"""
Intelligent Delegation System

This module implements advanced delegation logic that automatically selects the most
appropriate specialized agent for each task and coordinates parallel execution when
possible, avoiding race conditions and file conflicts.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Set, Optional, Any, Tuple
from pathlib import Path
import hashlib
import re

from ..agents import get_agent_loader
from ..quality import get_quality_gate_system, QualityGateResult

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be delegated."""
    CODE_IMPLEMENTATION = "code_implementation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DESIGN = "design"
    RESEARCH = "research"
    DOCUMENTATION = "documentation"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ARCHITECTURE_DESIGN = "architecture_design"
    PROJECT_MANAGEMENT = "project_management"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class Task:
    """Represents a task to be executed."""
    id: str
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    required_capabilities: List[str]
    affected_files: Set[str]
    dependencies: List[str]  # Task IDs this task depends on
    estimated_duration_minutes: int = 30
    can_run_parallel: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class AgentAssignment:
    """Represents an assignment of tasks to an agent."""
    agent_type: str
    tasks: List[Task]
    estimated_total_duration: int
    file_access_patterns: Set[str]


class TaskAnalyzer:
    """Analyzes tasks to determine type, requirements, and characteristics."""
    
    def __init__(self):
        self.agent_loader = get_agent_loader()
        self.agent_loader.load_all_descriptions()
        
        # Task classification patterns
        self.task_patterns = {
            TaskType.CODE_IMPLEMENTATION: [
                r'\b(implement|code|develop|build|create)\b',
                r'\b(function|class|method|api|endpoint)\b',
                r'\b(backend|frontend|database|service)\b'
            ],
            TaskType.CODE_REVIEW: [
                r'\b(review|audit|check|analyze)\b.*\b(code|implementation)\b',
                r'\b(pull request|PR|merge request|MR)\b'
            ],
            TaskType.TESTING: [
                r'\b(test|testing|unit test|integration test|e2e)\b',
                r'\b(qa|quality assurance|bug|defect)\b'
            ],
            TaskType.DESIGN: [
                r'\b(design|ui|ux|interface|wireframe|prototype)\b',
                r'\b(user experience|user interface|mockup)\b'
            ],
            TaskType.RESEARCH: [
                r'\b(research|analyze|investigate|study)\b',
                r'\b(market|user|competitive|trends)\b'
            ],
            TaskType.DOCUMENTATION: [
                r'\b(document|documentation|readme|guide|tutorial)\b',
                r'\b(api docs|user guide|manual)\b'
            ],
            TaskType.SECURITY_AUDIT: [
                r'\b(security|vulnerability|penetration|audit)\b',
                r'\b(encryption|authentication|authorization)\b'
            ],
            TaskType.PERFORMANCE_OPTIMIZATION: [
                r'\b(performance|optimize|speed|latency|throughput)\b',
                r'\b(scaling|load|benchmark|profiling)\b'
            ]
        }
    
    def analyze_task(self, title: str, description: str) -> Tuple[TaskType, List[str], Set[str]]:
        """Analyze a task to determine its type, capabilities needed, and affected files.
        
        Returns:
            Tuple of (task_type, required_capabilities, affected_files)
        """
        text = f"{title} {description}".lower()
        
        # Determine task type
        task_type = TaskType.CODE_IMPLEMENTATION  # Default
        max_matches = 0
        
        for ttype, patterns in self.task_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
            if matches > max_matches:
                max_matches = matches
                task_type = ttype
        
        # Extract required capabilities based on keywords
        capabilities = self._extract_capabilities(text)
        
        # Extract affected files
        affected_files = self._extract_file_patterns(text)
        
        return task_type, capabilities, affected_files
    
    def _extract_capabilities(self, text: str) -> List[str]:
        """Extract required capabilities from task description."""
        capability_keywords = {
            'backend development': ['backend', 'api', 'server', 'database'],
            'frontend development': ['frontend', 'ui', 'react', 'vue', 'angular'],
            'security analysis': ['security', 'vulnerability', 'auth', 'encryption'],
            'performance optimization': ['performance', 'optimization', 'speed', 'scaling'],
            'testing': ['test', 'qa', 'quality', 'bug'],
            'design': ['design', 'ui', 'ux', 'interface'],
            'research': ['research', 'analysis', 'market', 'user'],
            'documentation': ['documentation', 'docs', 'readme', 'guide']
        }
        
        capabilities = []
        for capability, keywords in capability_keywords.items():
            if any(keyword in text for keyword in keywords):
                capabilities.append(capability)
                
        return capabilities
    
    def _extract_file_patterns(self, text: str) -> Set[str]:
        """Extract file patterns that might be affected by the task."""
        file_patterns = set()
        
        # Look for explicit file mentions
        file_matches = re.findall(r'\b[\w/\\.-]+\.[\w]+\b', text)
        file_patterns.update(file_matches)
        
        # Infer file patterns from task context
        if 'backend' in text or 'api' in text:
            file_patterns.add('**/*.py')
        if 'frontend' in text or 'ui' in text:
            file_patterns.add('**/*.tsx')
            file_patterns.add('**/*.jsx') 
            file_patterns.add('**/*.css')
        if 'database' in text or 'schema' in text:
            file_patterns.add('**/*.sql')
            file_patterns.add('**/migrations/*')
        if 'config' in text or 'settings' in text:
            file_patterns.add('**/*config*')
            file_patterns.add('**/*settings*')
            
        return file_patterns


class AgentMatcher:
    """Matches tasks to the most appropriate agents."""
    
    def __init__(self):
        self.agent_loader = get_agent_loader()
        self.agent_loader.load_all_descriptions()
        
        # Agent specialization mapping
        self.task_to_agents = {
            TaskType.CODE_IMPLEMENTATION: [
                'backend_engineer', 'web_frontend_engineer', 'api_engineer',
                'mobile_engineer', 'database_engineer'
            ],
            TaskType.CODE_REVIEW: [
                'chief_qa_engineer', 'security_engineer', 'solutions_architect'
            ],
            TaskType.TESTING: [
                'backend_qa_engineer', 'web_frontend_qa_engineer',
                'tui_frontend_qa_engineer', 'chief_qa_engineer'
            ],
            TaskType.DESIGN: [
                'ux_ui_designer', 'tui_ux_designer', 'product_manager'
            ],
            TaskType.RESEARCH: [
                'user_researcher', 'market_researcher', 'data_analyst'
            ],
            TaskType.DOCUMENTATION: [
                'technical_writer', 'product_manager'
            ],
            TaskType.SECURITY_AUDIT: [
                'security_engineer'
            ],
            TaskType.PERFORMANCE_OPTIMIZATION: [
                'performance_engineer', 'site_reliability_engineer'
            ],
            TaskType.ARCHITECTURE_DESIGN: [
                'solutions_architect', 'backend_engineer'
            ],
            TaskType.PROJECT_MANAGEMENT: [
                'product_manager', 'business_analyst'
            ]
        }
    
    def find_best_agent(self, task: Task) -> str:
        """Find the best agent for a specific task.
        
        Args:
            task: The task to find an agent for.
            
        Returns:
            Agent type that best matches the task requirements.
        """
        # Get potential agents for this task type
        candidate_agents = self.task_to_agents.get(task.task_type, [])
        
        if not candidate_agents:
            # Fallback to general agents
            candidate_agents = ['backend_engineer', 'solutions_architect']
        
        # Score agents based on capability match
        best_agent = candidate_agents[0]
        best_score = 0
        
        for agent_type in candidate_agents:
            score = self._score_agent_for_task(agent_type, task)
            if score > best_score:
                best_score = score
                best_agent = agent_type
                
        return best_agent
    
    def _score_agent_for_task(self, agent_type: str, task: Task) -> float:
        """Score how well an agent matches a task."""
        agent_capabilities = self.agent_loader.get_agent_capabilities(agent_type)
        
        if not agent_capabilities:
            return 0.0
            
        # Calculate capability overlap
        capability_overlap = len(set(task.required_capabilities) & set(agent_capabilities))
        total_capabilities = len(task.required_capabilities) if task.required_capabilities else 1
        
        capability_score = capability_overlap / total_capabilities
        
        # Add bonus for perfect task type match
        primary_agents = self.task_to_agents.get(task.task_type, [])
        if agent_type in primary_agents[:2]:  # Top 2 agents for task type
            capability_score += 0.5
            
        return capability_score


class ConflictDetector:
    """Detects and resolves conflicts between parallel tasks."""
    
    def __init__(self):
        self.quality_gate = get_quality_gate_system()
    
    def detect_conflicts(self, assignments: List[AgentAssignment]) -> List[Tuple[str, str, str]]:
        """Detect conflicts between agent assignments.
        
        Returns:
            List of (agent1, agent2, conflict_reason) tuples.
        """
        conflicts = []
        
        for i, assignment1 in enumerate(assignments):
            for j, assignment2 in enumerate(assignments[i + 1:], i + 1):
                conflict_reason = self._check_assignment_conflict(assignment1, assignment2)
                if conflict_reason:
                    conflicts.append((assignment1.agent_type, assignment2.agent_type, conflict_reason))
                    
        return conflicts
    
    def _check_assignment_conflict(self, assignment1: AgentAssignment, assignment2: AgentAssignment) -> Optional[str]:
        """Check if two assignments conflict."""
        # File access conflicts
        file_overlap = assignment1.file_access_patterns & assignment2.file_access_patterns
        if file_overlap:
            # Check if both are writing to same files
            write_patterns = {pattern for pattern in file_overlap 
                            if not pattern.endswith('.md') and not pattern.endswith('.txt')}
            if write_patterns:
                return f"File write conflict: {write_patterns}"
        
        # Dependency conflicts (one agent depends on another's output)
        task1_ids = {task.id for task in assignment1.tasks}
        task2_ids = {task.id for task in assignment2.tasks}
        
        for task in assignment1.tasks:
            if any(dep in task2_ids for dep in task.dependencies):
                return "Task dependency conflict"
                
        for task in assignment2.tasks:
            if any(dep in task1_ids for dep in task.dependencies):
                return "Task dependency conflict"
        
        return None


class IntelligentDelegationSystem:
    """Main intelligent delegation coordinator."""
    
    def __init__(self):
        self.task_analyzer = TaskAnalyzer()
        self.agent_matcher = AgentMatcher()
        self.conflict_detector = ConflictDetector()
        self.agent_loader = get_agent_loader()
        
    async def delegate_tasks(self, task_descriptions: List[str]) -> List[AgentAssignment]:
        """Delegate a list of tasks to appropriate agents with optimal parallelization.
        
        Args:
            task_descriptions: List of task descriptions to delegate.
            
        Returns:
            List of agent assignments optimized for parallel execution.
        """
        # Step 1: Parse and analyze tasks
        tasks = []
        for i, desc in enumerate(task_descriptions):
            task_type, capabilities, files = self.task_analyzer.analyze_task(
                title=f"Task {i+1}", 
                description=desc
            )
            
            task = Task(
                id=f"task_{i}",
                title=f"Task {i+1}",
                description=desc,
                task_type=task_type,
                priority=TaskPriority.MEDIUM,
                required_capabilities=capabilities,
                affected_files=files,
                dependencies=[]
            )
            tasks.append(task)
        
        # Step 2: Match tasks to agents
        initial_assignments = self._create_initial_assignments(tasks)
        
        # Step 3: Optimize for parallel execution
        optimized_assignments = self._optimize_for_parallelism(initial_assignments)
        
        # Step 4: Validate and resolve conflicts
        final_assignments = await self._resolve_conflicts(optimized_assignments)
        
        return final_assignments
    
    def _create_initial_assignments(self, tasks: List[Task]) -> List[AgentAssignment]:
        """Create initial task-to-agent assignments."""
        agent_assignments = {}
        
        for task in tasks:
            best_agent = self.agent_matcher.find_best_agent(task)
            
            if best_agent not in agent_assignments:
                agent_assignments[best_agent] = AgentAssignment(
                    agent_type=best_agent,
                    tasks=[],
                    estimated_total_duration=0,
                    file_access_patterns=set()
                )
            
            assignment = agent_assignments[best_agent]
            assignment.tasks.append(task)
            assignment.estimated_total_duration += task.estimated_duration_minutes
            assignment.file_access_patterns.update(task.affected_files)
        
        return list(agent_assignments.values())
    
    def _optimize_for_parallelism(self, assignments: List[AgentAssignment]) -> List[AgentAssignment]:
        """Optimize assignments for maximum parallelism."""
        # Load balancing: split large assignments
        optimized = []
        
        for assignment in assignments:
            if len(assignment.tasks) > 3 and assignment.estimated_total_duration > 60:
                # Split large assignments
                mid = len(assignment.tasks) // 2
                
                assignment1 = AgentAssignment(
                    agent_type=assignment.agent_type,
                    tasks=assignment.tasks[:mid],
                    estimated_total_duration=sum(t.estimated_duration_minutes for t in assignment.tasks[:mid]),
                    file_access_patterns=set().union(*(t.affected_files for t in assignment.tasks[:mid]))
                )
                
                assignment2 = AgentAssignment(
                    agent_type=assignment.agent_type,
                    tasks=assignment.tasks[mid:],
                    estimated_total_duration=sum(t.estimated_duration_minutes for t in assignment.tasks[mid:]),
                    file_access_patterns=set().union(*(t.affected_files for t in assignment.tasks[mid:]))
                )
                
                optimized.extend([assignment1, assignment2])
            else:
                optimized.append(assignment)
        
        return optimized
    
    async def _resolve_conflicts(self, assignments: List[AgentAssignment]) -> List[AgentAssignment]:
        """Resolve conflicts between assignments."""
        conflicts = self.conflict_detector.detect_conflicts(assignments)
        
        if not conflicts:
            return assignments
        
        # For now, serialize conflicting assignments
        # TODO: Implement more sophisticated conflict resolution
        resolved = []
        conflicting_agents = set()
        
        for agent1, agent2, reason in conflicts:
            conflicting_agents.update([agent1, agent2])
            logger.warning(f"Conflict between {agent1} and {agent2}: {reason}")
        
        # Create execution groups
        parallel_group = []
        sequential_group = []
        
        for assignment in assignments:
            if assignment.agent_type in conflicting_agents:
                sequential_group.append(assignment)
            else:
                parallel_group.append(assignment)
        
        # Return parallel assignments first, then sequential ones
        return parallel_group + sequential_group
    
    def suggest_task_breakdown(self, complex_task: str) -> List[str]:
        """Break down a complex task into smaller, parallelizable subtasks."""
        # This is a simplified version - could be enhanced with LLM assistance
        subtasks = []
        
        if "implement" in complex_task.lower():
            subtasks.extend([
                f"Design architecture for: {complex_task}",
                f"Implement core functionality: {complex_task}",
                f"Add tests for: {complex_task}",
                f"Document implementation: {complex_task}"
            ])
        elif "review" in complex_task.lower():
            subtasks.extend([
                f"Security review: {complex_task}",
                f"Code quality review: {complex_task}",
                f"Performance review: {complex_task}"
            ])
        else:
            # Default breakdown
            subtasks = [complex_task]
        
        return subtasks


# Global delegation system
_delegation_system = None


def get_delegation_system() -> IntelligentDelegationSystem:
    """Get the global intelligent delegation system instance."""
    global _delegation_system
    if _delegation_system is None:
        _delegation_system = IntelligentDelegationSystem()
    return _delegation_system