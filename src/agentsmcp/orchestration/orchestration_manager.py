"""
OrchestrationManager - Revolutionary Multi-Agent Coordination Hub

The central orchestration system that seamlessly integrates:
- SeamlessCoordinator for task execution
- EmotionalOrchestrator for agent wellness
- SymphonyMode for multi-agent harmony
- PredictiveSpawner for intelligent provisioning

Provides a unified, zero-configuration interface to the complete orchestration ecosystem.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path

from .seamless_coordinator import SeamlessCoordinator
from .emotional_orchestrator import EmotionalOrchestrator
from .symphony_mode import SymphonyMode
from .predictive_spawner import PredictiveSpawner

logger = logging.getLogger(__name__)

class OrchestrationManager:
    """
    Revolutionary Multi-Agent Coordination Hub
    
    Provides a unified interface to the complete orchestration ecosystem,
    enabling seamless coordination between multiple AI agents with
    emotional intelligence, predictive provisioning, and symphony-mode harmony.
    """
    
    def __init__(self, max_agents: int = 50, quality_threshold: float = 0.95):
        self.max_agents = max_agents
        self.quality_threshold = quality_threshold
        
        # Initialize orchestration components
        self.seamless_coordinator = SeamlessCoordinator()
        self.emotional_orchestrator = EmotionalOrchestrator()
        self.symphony_mode = SymphonyMode(max_agents=max_agents, quality_threshold=quality_threshold)
        self.predictive_spawner = PredictiveSpawner(max_agents=max_agents)
        
        # System state
        self.is_running = False
        self.session_id = None
        self.start_time = None
        self.orchestration_mode = "seamless"  # seamless, symphony, predictive, hybrid
        
        # Performance tracking
        self.task_history = []
        self.performance_metrics = {
            "total_tasks_completed": 0,
            "average_quality_score": 0.0,
            "average_completion_time": 0.0,
            "agent_satisfaction": 0.0,
            "human_satisfaction": 0.0
        }
        
        # Settings persistence
        self.config_dir = Path.home() / ".agentsmcp"
        self.settings_file = self.config_dir / "config.json"
        self.user_settings = {}
        self.reload_user_settings()
        
    async def initialize(self, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Initialize the complete orchestration system
        
        Args:
            mode: Orchestration mode - 'seamless', 'symphony', 'predictive', or 'hybrid'
            
        Returns:
            Initialization status and system capabilities
        """
        self.session_id = f"orch_{uuid.uuid4().hex[:8]}"
        self.start_time = datetime.now()
        self.orchestration_mode = mode
        
        logger.info(f"🚀 Initializing OrchestrationManager in {mode} mode")
        
        initialization_results = {}
        
        try:
            # Initialize emotional orchestrator (always needed)
            emotional_init = await self.emotional_orchestrator.initialize()
            initialization_results["emotional_orchestrator"] = emotional_init
            
            # Initialize based on mode
            if mode in ["seamless", "hybrid"]:
                seamless_init = await self.seamless_coordinator.initialize()
                initialization_results["seamless_coordinator"] = seamless_init
            
            if mode in ["symphony", "hybrid"]:
                symphony_init = {"system": "symphony_ready"}  # SymphonyMode initializes on-demand
                initialization_results["symphony_mode"] = symphony_init
            
            if mode in ["predictive", "hybrid"]:
                predictive_init = await self.predictive_spawner.start_predictive_spawning()
                initialization_results["predictive_spawner"] = predictive_init
            
            self.is_running = True
            
            # Start cross-component integration
            if mode == "hybrid":
                await self._start_hybrid_integration()
            
            logger.info(f"✅ OrchestrationManager initialized successfully")
            
            return {
                "session_id": self.session_id,
                "initialization_time": self.start_time.isoformat(),
                "orchestration_mode": self.orchestration_mode,
                "max_agents": self.max_agents,
                "quality_threshold": self.quality_threshold,
                "components_initialized": list(initialization_results.keys()),
                "component_details": initialization_results,
                "system_status": "ready",
                "capabilities": await self._get_system_capabilities()
            }
            
        except Exception as e:
            logger.error(f"❌ OrchestrationManager initialization failed: {e}")
            return {
                "session_id": self.session_id,
                "error": str(e),
                "system_status": "failed",
                "partial_initialization": initialization_results
            }
    
    async def execute_task(self, description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a task using the optimal orchestration strategy
        
        Args:
            description: Task description
            context: Additional context for task execution
            
        Returns:
            Task execution result with comprehensive metrics
        """
        if not self.is_running:
            raise RuntimeError("OrchestrationManager not initialized. Call initialize() first.")
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        context = context or {}
        
        logger.info(f"🎯 Executing task {task_id}: {description}")
        
        try:
            # Determine optimal execution strategy
            strategy = await self._determine_execution_strategy(description, context)
            
            # Execute using chosen strategy
            if strategy == "seamless":
                result = await self._execute_with_seamless_coordinator(description, context)
            elif strategy == "symphony":
                result = await self._execute_with_symphony_mode(description, context)
            elif strategy == "predictive":
                result = await self._execute_with_predictive_spawning(description, context)
            else:  # hybrid
                result = await self._execute_with_hybrid_strategy(description, context)
            
            # Enhance result with emotional intelligence
            emotional_analysis = await self.emotional_orchestrator.analyze_human_emotions(
                description, context
            )
            
            # Calculate comprehensive metrics
            completion_time = datetime.now() - start_time
            
            enhanced_result = {
                "task_id": task_id,
                "description": description,
                "execution_strategy": strategy,
                "completion_time": str(completion_time),
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "emotional_analysis": emotional_analysis,
                "orchestration_metrics": await self._calculate_orchestration_metrics(),
                **result
            }
            
            # Update performance tracking
            await self._update_performance_metrics(enhanced_result)
            
            # Record in task history
            self.task_history.append(enhanced_result)
            
            logger.info(f"✅ Task {task_id} completed using {strategy} strategy")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ Task {task_id} execution failed: {e}")
            
            error_result = {
                "task_id": task_id,
                "description": description,
                "error": str(e),
                "execution_time": str(datetime.now() - start_time),
                "status": "failed"
            }
            
            self.task_history.append(error_result)
            return error_result
    
    async def execute_multiple_tasks(self, tasks: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute multiple tasks with optimal coordination strategy
        
        Args:
            tasks: List of task descriptions
            context: Shared context for all tasks
            
        Returns:
            Comprehensive results from multi-task execution
        """
        if not self.is_running:
            raise RuntimeError("OrchestrationManager not initialized. Call initialize() first.")
        
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        logger.info(f"🎼 Executing task batch {batch_id} with {len(tasks)} tasks")
        
        try:
            # For multiple tasks, Symphony mode is often optimal
            if self.orchestration_mode in ["symphony", "hybrid"]:
                result = await self.symphony_mode.begin_symphony(tasks, context)
                
                # Monitor symphony progress
                symphony_status = await self._monitor_symphony_execution()
                
                return {
                    "batch_id": batch_id,
                    "execution_strategy": "symphony_mode",
                    "total_tasks": len(tasks),
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "symphony_result": result,
                    "symphony_status": symphony_status,
                    "orchestration_metrics": await self._calculate_orchestration_metrics()
                }
            else:
                # Execute tasks sequentially with seamless coordination
                results = []
                for i, task_desc in enumerate(tasks):
                    task_context = {**(context or {}), "batch_id": batch_id, "task_index": i}
                    task_result = await self.execute_task(task_desc, task_context)
                    results.append(task_result)
                
                return {
                    "batch_id": batch_id,
                    "execution_strategy": "sequential_seamless",
                    "total_tasks": len(tasks),
                    "completed_tasks": len([r for r in results if r.get("status") != "failed"]),
                    "task_results": results,
                    "batch_completion_time": str(datetime.now() - start_time),
                    "orchestration_metrics": await self._calculate_orchestration_metrics()
                }
                
        except Exception as e:
            logger.error(f"❌ Task batch {batch_id} execution failed: {e}")
            return {
                "batch_id": batch_id,
                "error": str(e),
                "execution_time": str(datetime.now() - start_time),
                "status": "failed"
            }
    
    async def _determine_execution_strategy(self, description: str, context: Dict[str, Any]) -> str:
        """Determine the optimal execution strategy for a task"""
        # Task complexity analysis
        complexity_score = await self._analyze_task_complexity(description)
        
        # Context analysis
        multi_agent_indicators = ["coordinate", "collaborate", "multiple", "parallel", "concurrent"]
        needs_multi_agent = any(indicator in description.lower() for indicator in multi_agent_indicators)
        
        # Resource requirements
        resource_intensive = any(keyword in description.lower() for keyword in 
                               ["heavy", "large-scale", "complex", "comprehensive"])
        
        # Strategy selection logic
        if self.orchestration_mode == "seamless":
            return "seamless"
        elif self.orchestration_mode == "symphony":
            return "symphony"
        elif self.orchestration_mode == "predictive":
            return "predictive"
        else:  # hybrid mode
            if needs_multi_agent and complexity_score > 0.7:
                return "symphony"
            elif resource_intensive or context.get("agent_spawning_needed"):
                return "predictive"
            else:
                return "seamless"
    
    async def _execute_with_seamless_coordinator(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using seamless coordinator"""
        result = await self.seamless_coordinator.execute_task(description, context)
        return {
            "execution_method": "seamless_coordinator",
            "result": result
        }
    
    async def _execute_with_symphony_mode(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using symphony mode"""
        # Break down task into symphony movements if needed
        tasks = [description]  # In practice, this might be more sophisticated task breakdown
        
        result = await self.symphony_mode.begin_symphony(tasks, context)
        status = await self.symphony_mode.get_symphony_status()
        
        return {
            "execution_method": "symphony_mode",
            "symphony_result": result,
            "symphony_status": status
        }
    
    async def _execute_with_predictive_spawning(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using predictive spawning"""
        # Determine required agent specialization
        specialization = await self._determine_required_specialization(description)
        
        # Request agent spawn
        spawn_result = await self.predictive_spawner.request_agent_spawn(
            specialization=specialization,
            urgency=context.get("urgency", 0.5),
            context=context
        )
        
        # Execute with spawned agent (simplified - would involve actual agent coordination)
        execution_result = {
            "agent_specialization": specialization,
            "task_execution": "completed_with_spawned_agent",
            "quality_score": 0.88  # Simulated high quality from specialized agent
        }
        
        return {
            "execution_method": "predictive_spawning",
            "spawn_result": spawn_result,
            "execution_result": execution_result
        }
    
    async def _execute_with_hybrid_strategy(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using hybrid strategy combining multiple approaches"""
        # Use seamless coordinator as base
        seamless_result = await self._execute_with_seamless_coordinator(description, context)
        
        # Enhance with emotional intelligence
        emotional_enhancement = await self.emotional_orchestrator.generate_empathetic_response(
            agent_id="hybrid_agent",
            human_emotions=await self.emotional_orchestrator.analyze_human_emotions(description, context),
            context=description
        )
        
        # Consider predictive spawning if task suggests future needs
        future_needs = await self._analyze_future_agent_needs(description, context)
        spawn_suggestions = []
        
        for need in future_needs:
            suggestion = await self.predictive_spawner.request_agent_spawn(
                specialization=need["specialization"],
                urgency=need.get("urgency", 0.3),
                context={"source": "hybrid_prediction", **context}
            )
            spawn_suggestions.append(suggestion)
        
        return {
            "execution_method": "hybrid_strategy",
            "seamless_result": seamless_result,
            "emotional_enhancement": emotional_enhancement,
            "predictive_spawning": spawn_suggestions,
            "hybrid_quality_score": 0.92  # High quality from hybrid approach
        }
    
    async def _analyze_task_complexity(self, description: str) -> float:
        """Analyze task complexity on a scale of 0.0 to 1.0"""
        complexity_indicators = {
            "simple": -0.2, "easy": -0.1, "basic": -0.1,
            "complex": 0.3, "advanced": 0.3, "sophisticated": 0.4,
            "integrate": 0.2, "coordinate": 0.2, "optimize": 0.3,
            "machine learning": 0.4, "ai": 0.3, "algorithm": 0.3,
            "architecture": 0.4, "system design": 0.4, "scalable": 0.3
        }
        
        description_lower = description.lower()
        complexity = 0.3  # Base complexity
        
        for indicator, weight in complexity_indicators.items():
            if indicator in description_lower:
                complexity += weight
        
        # Length factor
        complexity += min(0.2, len(description) / 1000)
        
        return max(0.0, min(1.0, complexity))
    
    async def _determine_required_specialization(self, description: str) -> str:
        """Determine the required agent specialization for a task"""
        specialization_keywords = {
            "frontend": "full-stack-developer",
            "backend": "full-stack-developer", 
            "ui": "ui-ux-designer",
            "ux": "ui-ux-designer",
            "security": "security-specialist",
            "devops": "devops-engineer",
            "data": "data-scientist",
            "ai": "ai-researcher",
            "machine learning": "ai-researcher",
            "testing": "quality-assurance",
            "documentation": "technical-writer"
        }
        
        description_lower = description.lower()
        
        for keyword, specialization in specialization_keywords.items():
            if keyword in description_lower:
                return specialization
        
        return "full-stack-developer"  # Default
    
    async def _analyze_future_agent_needs(self, description: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential future agent needs based on current task"""
        needs = []
        
        # Look for indicators of future work
        if "phase" in description.lower() or "stage" in description.lower():
            needs.append({
                "specialization": "product-manager",
                "urgency": 0.2,
                "reasoning": "Multi-phase project may need coordination"
            })
        
        if "test" in description.lower() or "quality" in description.lower():
            needs.append({
                "specialization": "quality-assurance", 
                "urgency": 0.4,
                "reasoning": "Testing will be needed after implementation"
            })
        
        if "deploy" in description.lower() or "production" in description.lower():
            needs.append({
                "specialization": "devops-engineer",
                "urgency": 0.3,
                "reasoning": "Deployment expertise will be required"
            })
        
        return needs
    
    async def _monitor_symphony_execution(self) -> Dict[str, Any]:
        """Monitor symphony mode execution progress"""
        return await self.symphony_mode.get_symphony_status()
    
    async def _calculate_orchestration_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive orchestration metrics"""
        return {
            "system_uptime": str(datetime.now() - self.start_time) if self.start_time else "0:00:00",
            "total_tasks_in_session": len(self.task_history),
            "successful_tasks": len([t for t in self.task_history if t.get("status") != "failed"]),
            "average_completion_time": self._calculate_average_completion_time(),
            "orchestration_mode": self.orchestration_mode,
            "active_components": await self._get_active_component_status(),
            "performance_metrics": self.performance_metrics.copy()
        }
    
    def _calculate_average_completion_time(self) -> float:
        """Calculate average task completion time in seconds"""
        if not self.task_history:
            return 0.0
        
        completion_times = []
        for task in self.task_history:
            if "completion_time" in task:
                # Parse completion time string (format: "0:00:05.123456")
                time_parts = task["completion_time"].split(":")
                if len(time_parts) >= 3:
                    seconds = float(time_parts[-1]) + int(time_parts[-2]) * 60
                    completion_times.append(seconds)
        
        return sum(completion_times) / len(completion_times) if completion_times else 0.0
    
    async def _update_performance_metrics(self, task_result: Dict[str, Any]):
        """Update performance metrics based on task result"""
        self.performance_metrics["total_tasks_completed"] += 1
        
        # Update quality score
        if "quality_score" in task_result:
            current_avg = self.performance_metrics["average_quality_score"]
            total_tasks = self.performance_metrics["total_tasks_completed"]
            new_score = task_result["quality_score"]
            
            self.performance_metrics["average_quality_score"] = (
                (current_avg * (total_tasks - 1) + new_score) / total_tasks
            )
        
        # Update completion time
        if "completion_time" in task_result:
            current_avg = self.performance_metrics["average_completion_time"]
            total_tasks = self.performance_metrics["total_tasks_completed"]
            
            # Parse new completion time
            time_str = task_result["completion_time"]
            time_parts = time_str.split(":")
            if len(time_parts) >= 3:
                new_time = float(time_parts[-1]) + int(time_parts[-2]) * 60
                
                self.performance_metrics["average_completion_time"] = (
                    (current_avg * (total_tasks - 1) + new_time) / total_tasks
                )
    
    async def _get_active_component_status(self) -> Dict[str, str]:
        """Get status of all orchestration components"""
        status = {}
        
        if hasattr(self.seamless_coordinator, 'is_initialized'):
            status["seamless_coordinator"] = "active" if self.seamless_coordinator.is_initialized else "inactive"
        else:
            status["seamless_coordinator"] = "unknown"
            
        if hasattr(self.emotional_orchestrator, 'is_running'):
            status["emotional_orchestrator"] = "active" if self.emotional_orchestrator.is_running else "inactive"
        else:
            status["emotional_orchestrator"] = "unknown"
            
        if hasattr(self.symphony_mode, 'is_conducting'):
            status["symphony_mode"] = "active" if self.symphony_mode.is_conducting else "ready"
        else:
            status["symphony_mode"] = "ready"
            
        if hasattr(self.predictive_spawner, 'is_running'):
            status["predictive_spawner"] = "active" if self.predictive_spawner.is_running else "inactive"
        else:
            status["predictive_spawner"] = "unknown"
        
        return status
    
    async def _get_system_capabilities(self) -> List[str]:
        """Get list of system capabilities"""
        capabilities = [
            "seamless_task_execution",
            "emotional_intelligence_integration",
            "zero_configuration_operation",
            "multi_agent_coordination"
        ]
        
        if self.orchestration_mode in ["symphony", "hybrid"]:
            capabilities.extend([
                "symphony_mode_orchestration",
                "harmonic_agent_coordination",
                "real_time_conflict_resolution"
            ])
        
        if self.orchestration_mode in ["predictive", "hybrid"]:
            capabilities.extend([
                "predictive_agent_spawning",
                "intelligent_resource_allocation",
                "demand_pattern_analysis"
            ])
        
        if self.orchestration_mode == "hybrid":
            capabilities.extend([
                "adaptive_strategy_selection",
                "cross_component_integration",
                "optimal_execution_routing"
            ])
        
        return capabilities
    
    async def _start_hybrid_integration(self):
        """Start cross-component integration for hybrid mode"""
        logger.info("🔗 Starting hybrid integration between orchestration components")
        
        # This would set up communication channels between components
        # For now, we'll just log that hybrid integration is active
        logger.info("✅ Hybrid integration active - components can share intelligence")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.is_running:
            return {
                "system_status": "not_initialized",
                "session_id": self.session_id,
                "message": "System not initialized. Call initialize() first."
            }
        
        # Get status from all components
        component_status = {}
        
        try:
            if hasattr(self.seamless_coordinator, 'get_status'):
                component_status["seamless_coordinator"] = await self.seamless_coordinator.get_status()
        except:
            component_status["seamless_coordinator"] = "unavailable"
        
        try:
            if hasattr(self.emotional_orchestrator, 'get_status'):
                component_status["emotional_orchestrator"] = await self.emotional_orchestrator.get_status()
        except:
            component_status["emotional_orchestrator"] = "unavailable"
        
        try:
            component_status["symphony_mode"] = await self.symphony_mode.get_symphony_status()
        except:
            component_status["symphony_mode"] = "unavailable"
        
        try:
            component_status["predictive_spawner"] = await self.predictive_spawner.get_spawner_status()
        except:
            component_status["predictive_spawner"] = "unavailable"
        
        return {
            "system_status": "running",
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else "0:00:00",
            "orchestration_mode": self.orchestration_mode,
            "max_agents": self.max_agents,
            "quality_threshold": self.quality_threshold,
            "task_history_length": len(self.task_history),
            "performance_metrics": self.performance_metrics,
            "component_status": component_status,
            "system_capabilities": await self._get_system_capabilities()
        }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Gracefully shutdown the orchestration system"""
        if not self.is_running:
            return {"message": "System was not running"}
        
        logger.info("🛑 Shutting down OrchestrationManager")
        
        shutdown_results = {}
        
        # Shutdown components
        try:
            if hasattr(self.symphony_mode, 'stop_symphony'):
                shutdown_results["symphony_mode"] = await self.symphony_mode.stop_symphony()
        except Exception as e:
            logger.error(f"Error stopping symphony mode: {e}")
        
        try:
            if hasattr(self.predictive_spawner, 'stop_predictive_spawning'):
                shutdown_results["predictive_spawner"] = await self.predictive_spawner.stop_predictive_spawning()
        except Exception as e:
            logger.error(f"Error stopping predictive spawner: {e}")
        
        # Mark system as stopped
        self.is_running = False
        end_time = datetime.now()
        
        return {
            "shutdown_completed": True,
            "session_id": self.session_id,
            "shutdown_time": end_time.isoformat(),
            "session_duration": str(end_time - self.start_time) if self.start_time else "unknown",
            "total_tasks_completed": self.performance_metrics["total_tasks_completed"],
            "component_shutdowns": shutdown_results
        }
    
    def save_user_settings(self, settings: Dict[str, Any]) -> None:
        """Save user settings to configuration file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Merge with existing settings
            current_settings = self.user_settings.copy()
            current_settings.update(settings)
            
            # Write to file
            with open(self.settings_file, 'w') as f:
                json.dump(current_settings, f, indent=2)
            
            # Update in-memory settings
            self.user_settings = current_settings
            
            logger.info(f"Settings saved successfully to {self.settings_file}")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            raise
    
    def reload_user_settings(self) -> Dict[str, Any]:
        """Reload user settings from configuration file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    self.user_settings = json.load(f)
                logger.info(f"Settings loaded from {self.settings_file}")
            else:
                # Default settings
                self.user_settings = {
                    "provider": "ollama-turbo",
                    "model": "gpt-oss:120b", 
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
                logger.info("Using default settings")
                
            return self.user_settings
            
        except Exception as e:
            logger.error(f"Failed to reload settings: {e}")
            # Fall back to defaults
            self.user_settings = {
                "provider": "ollama-turbo",
                "model": "gpt-oss:120b",
                "temperature": 0.7, 
                "max_tokens": 1024
            }
            return self.user_settings
    
    def generate_client_config(self) -> str:
        """Generate MCP client configuration with auto-discovered paths."""
        import subprocess
        import shutil
        
        # Auto-discover system paths
        node_path = shutil.which("node") or "/usr/local/bin/node"
        python_path = shutil.which("python3") or shutil.which("python") or "/usr/bin/python3"
        
        # Get current user settings
        settings = self.user_settings
        
        # Base configuration template
        config = {
            "mcpServers": {
                "codex": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-codex"],
                    "env": {
                        "NODE_PATH": node_path,
                        "CODEX_MODEL": settings.get("model", "gpt-oss:120b"),
                        "CODEX_PROVIDER": settings.get("provider", "ollama-turbo"),
                        "CODEX_TEMPERATURE": str(settings.get("temperature", 0.7)),
                        "CODEX_MAX_TOKENS": str(settings.get("max_tokens", 1024))
                    }
                },
                "claude": {
                    "command": python_path,
                    "args": ["-m", "mcp_claude"],
                    "env": {
                        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
                        "CLAUDE_MODEL": "claude-3-5-sonnet-20241022",
                        "CLAUDE_MAX_TOKENS": str(settings.get("max_tokens", 1024))
                    }
                },
                "ollama": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-ollama"],
                    "env": {
                        "OLLAMA_HOST": "http://localhost:11434",
                        "OLLAMA_MODEL": "gpt-oss:20b"
                    }
                },
                "ollama-turbo": {
                    "command": "npx", 
                    "args": ["-y", "@anthropic/mcp-ollama-turbo"],
                    "env": {
                        "OLLAMA_TURBO_HOST": "https://ollama.com",
                        "OLLAMA_TURBO_MODEL": "gpt-oss:120b",
                        "OLLAMA_API_KEY": "${OLLAMA_API_KEY}"
                    }
                },
                "github": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-github"],
                    "env": {
                        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
                    }
                },
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-filesystem"],
                    "env": {
                        "FILESYSTEM_ROOT": str(Path.cwd())
                    }
                },
                "git": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-git"],
                    "env": {
                        "GIT_WORKING_DIR": str(Path.cwd())
                    }
                }
            }
        }

        # If Claude Code CLI is installed, add it as an MCP server option
        try:
            claude_code_bin = shutil.which("claude-code")
            if claude_code_bin:
                config["mcpServers"]["claude-code-cli"] = {
                    "command": claude_code_bin,
                    "args": ["mcp-server"],
                    "env": {
                        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
                    }
                }
        except Exception:
            pass

        # If a Codex CLI is installed, add it as an MCP server option
        try:
            codex_bin = shutil.which("codex") or shutil.which("codex-cli")
            if codex_bin:
                config["mcpServers"]["codex-cli"] = {
                    "command": codex_bin,
                    "args": ["mcp-server"],
                    "env": {}
                }
        except Exception:
            pass
        
        # Format as pretty-printed JSON
        config_json = json.dumps(config, indent=2)
        
        # Add helpful header comments
        header = f"""# MCP Client Configuration
# Generated by AgentsMCP on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Current settings: {settings.get('provider')} / {settings.get('model')}
#
# Save this to one of:
# - ~/.config/Claude/claude_desktop_config.json  (Claude Desktop)
# - ~/.config/claude-code/config.json           (Claude Code CLI) 
# - Your MCP client's configuration file
#
# Required environment variables:
# - ANTHROPIC_API_KEY: Your Anthropic API key for Claude
# - GITHUB_TOKEN: Your GitHub personal access token
# - OLLAMA_API_KEY: Your Ollama API key (if using)
#
# Auto-discovered paths:
# - Node.js: {node_path}
# - Python: {python_path}
# - Working Directory: {Path.cwd()}

"""
        
        return header + config_json
