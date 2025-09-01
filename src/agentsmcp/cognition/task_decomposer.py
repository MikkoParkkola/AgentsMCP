"""
Intelligent task breakdown system with dependency analysis and parallel execution opportunities.

This module implements sophisticated task decomposition strategies that identify
dependencies, parallelization opportunities, and optimal execution sequences.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from datetime import timedelta

from .models import (
    Approach, SubTask, DependencyGraph, TaskType, DecompositionStrategy
)
from .config import TaskDecomposerConfig, DEFAULT_DECOMPOSER_CONFIG

logger = logging.getLogger(__name__)


class TaskTooComplex(Exception):
    """Raised when task is too complex for decomposition."""
    pass


class CircularDependency(Exception):
    """Raised when circular dependencies are detected."""
    pass


class DecompositionTimeout(Exception):
    """Raised when decomposition takes too long."""
    pass


class TaskDecomposer:
    """
    Intelligent task breakdown system with dependency analysis.
    
    This decomposer analyzes approaches and breaks them down into manageable
    sub-tasks while identifying dependencies and parallel execution opportunities.
    """
    
    def __init__(self, config: Optional[TaskDecomposerConfig] = None):
        """Initialize the task decomposer."""
        self.config = config or DEFAULT_DECOMPOSER_CONFIG
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Decomposition metrics
        self._total_decompositions = 0
        self._total_subtasks_created = 0
        self._circular_dependencies_detected = 0
        
        # Task type classification patterns
        self._task_type_patterns = self._build_task_type_patterns()
        
        # Dependency detection patterns
        self._dependency_patterns = self._build_dependency_patterns()
        
        self.logger.info("TaskDecomposer initialized")
    
    async def decompose_approach(
        self,
        approach: Approach,
        config: Optional[TaskDecomposerConfig] = None
    ) -> Tuple[List[SubTask], DependencyGraph]:
        """
        Decompose an approach into sub-tasks with dependency analysis.
        
        Args:
            approach: The approach to decompose
            config: Optional configuration override
            
        Returns:
            Tuple of (subtasks, dependency_graph)
            
        Raises:
            TaskTooComplex: If approach is too complex
            CircularDependency: If circular dependencies are detected
            DecompositionTimeout: If decomposition takes too long
        """
        decomp_config = config or self.config
        
        # Check complexity threshold
        complexity = self._assess_complexity(approach)
        if complexity < decomp_config.min_task_complexity_for_decomposition:
            # Return single task for simple approaches
            single_task = await self._create_single_task_from_approach(approach)
            dependency_graph = DependencyGraph()
            dependency_graph.add_task(single_task)
            return [single_task], dependency_graph
        
        try:
            self._total_decompositions += 1
            
            # Initial decomposition based on approach steps
            subtasks = await self._decompose_steps(approach, decomp_config)
            
            # Enhance decomposition with additional analysis
            subtasks = await self._enhance_decomposition(subtasks, approach, decomp_config)
            
            # Build dependency graph
            dependency_graph = await self._build_dependency_graph(subtasks, decomp_config)
            
            # Detect and resolve circular dependencies
            if decomp_config.detect_circular_dependencies:
                await self._resolve_circular_dependencies(dependency_graph)
            
            # Optimize for parallel execution
            if decomp_config.enable_parallel_task_identification:
                await self._optimize_for_parallelization(dependency_graph)
            
            self._total_subtasks_created += len(subtasks)
            
            self.logger.info(
                f"Decomposed approach '{approach.name}' into {len(subtasks)} subtasks "
                f"with {len(dependency_graph.edges)} dependency relationships"
            )
            
            return subtasks, dependency_graph
            
        except Exception as e:
            self.logger.error(f"Error decomposing approach: {e}", exc_info=True)
            raise
    
    async def _decompose_steps(
        self,
        approach: Approach,
        config: TaskDecomposerConfig
    ) -> List[SubTask]:
        """Decompose approach steps into sub-tasks."""
        subtasks = []
        
        for i, step in enumerate(approach.steps):
            # Classify task type
            task_type = self._classify_task_type(step)
            
            # Estimate duration based on step complexity
            estimated_duration = self._estimate_step_duration(step, task_type)
            
            # Determine priority
            priority = self._calculate_priority(step, i, task_type)
            
            # Create sub-task
            subtask = SubTask(
                name=f"Step {i+1}: {step[:50]}",
                description=step,
                task_type=task_type,
                priority=priority,
                estimated_duration=estimated_duration,
                metadata={
                    "original_step_index": i,
                    "approach_id": approach.id,
                    "step_text": step
                }
            )
            
            subtasks.append(subtask)
        
        return subtasks
    
    async def _enhance_decomposition(
        self,
        subtasks: List[SubTask],
        approach: Approach,
        config: TaskDecomposerConfig
    ) -> List[SubTask]:
        """Enhance decomposition with additional analysis."""
        enhanced_subtasks = []
        
        for subtask in subtasks:
            # Check if subtask needs further decomposition
            if await self._should_further_decompose(subtask, config):
                sub_subtasks = await self._decompose_subtask(subtask, config)
                enhanced_subtasks.extend(sub_subtasks)
            else:
                enhanced_subtasks.append(subtask)
        
        # Add setup and validation tasks if needed
        enhanced_subtasks = await self._add_implicit_tasks(enhanced_subtasks, approach, config)
        
        return enhanced_subtasks
    
    async def _build_dependency_graph(
        self,
        subtasks: List[SubTask],
        config: TaskDecomposerConfig
    ) -> DependencyGraph:
        """Build dependency graph from sub-tasks."""
        dependency_graph = DependencyGraph()
        
        # Add all tasks to graph
        for task in subtasks:
            dependency_graph.add_task(task)
        
        if not config.enable_dependency_detection:
            return dependency_graph
        
        # Detect dependencies between tasks
        for i, task_a in enumerate(subtasks):
            for j, task_b in enumerate(subtasks):
                if i != j and await self._has_dependency(task_a, task_b):
                    dependency_graph.add_dependency(task_b.id, task_a.id)
                    # Also update task's dependency list
                    if task_a.id not in task_b.dependencies:
                        task_b.dependencies.append(task_a.id)
        
        return dependency_graph
    
    async def _resolve_circular_dependencies(self, dependency_graph: DependencyGraph):
        """Detect and resolve circular dependencies."""
        cycles = dependency_graph.detect_cycles()
        
        if cycles:
            self._circular_dependencies_detected += len(cycles)
            self.logger.warning(f"Detected {len(cycles)} circular dependencies")
            
            for cycle in cycles:
                await self._resolve_cycle(cycle, dependency_graph)
    
    async def _resolve_cycle(self, cycle: List[str], dependency_graph: DependencyGraph):
        """Resolve a specific circular dependency cycle."""
        if len(cycle) <= 2:
            return  # Can't resolve trivial cycles
        
        # Strategy: Break the weakest dependency in the cycle
        weakest_link = await self._find_weakest_dependency_in_cycle(cycle, dependency_graph)
        
        if weakest_link:
            from_task, to_task = weakest_link
            # Remove the dependency
            if to_task in dependency_graph.edges.get(from_task, []):
                dependency_graph.edges[from_task].remove(to_task)
                
            # Update task dependency list
            if from_task in dependency_graph.tasks[to_task].dependencies:
                dependency_graph.tasks[to_task].dependencies.remove(from_task)
            
            self.logger.info(f"Resolved circular dependency by removing link: {from_task} -> {to_task}")
    
    async def _optimize_for_parallelization(self, dependency_graph: DependencyGraph):
        """Optimize dependency graph for parallel execution opportunities."""
        # Identify tasks that can be merged for efficiency
        merge_candidates = await self._find_merge_candidates(dependency_graph)
        
        for candidate_group in merge_candidates:
            await self._merge_tasks(candidate_group, dependency_graph)
        
        # Identify tasks that should be split for better parallelization
        split_candidates = await self._find_split_candidates(dependency_graph)
        
        for candidate in split_candidates:
            await self._split_task_for_parallelization(candidate, dependency_graph)
    
    def _classify_task_type(self, step_text: str) -> TaskType:
        """Classify a task step into a TaskType."""
        step_lower = step_text.lower()
        
        # Check each task type's keywords
        for task_type, keywords in self._task_type_patterns.items():
            if any(keyword in step_lower for keyword in keywords):
                return task_type
        
        # Default to implementation
        return TaskType.IMPLEMENTATION
    
    def _estimate_step_duration(self, step_text: str, task_type: TaskType) -> timedelta:
        """Estimate duration for a step based on complexity indicators."""
        base_durations = {
            TaskType.ANALYSIS: timedelta(minutes=30),
            TaskType.IMPLEMENTATION: timedelta(hours=2),
            TaskType.TESTING: timedelta(minutes=45),
            TaskType.DOCUMENTATION: timedelta(minutes=20),
            TaskType.COORDINATION: timedelta(minutes=15),
            TaskType.VALIDATION: timedelta(minutes=30)
        }
        
        base_duration = base_durations.get(task_type, timedelta(hours=1))
        
        # Adjust based on complexity indicators
        complexity_multiplier = 1.0
        
        complexity_indicators = ['complex', 'comprehensive', 'detailed', 'thorough', 'complete']
        simple_indicators = ['simple', 'quick', 'basic', 'minimal']
        
        step_lower = step_text.lower()
        
        for indicator in complexity_indicators:
            if indicator in step_lower:
                complexity_multiplier += 0.5
        
        for indicator in simple_indicators:
            if indicator in step_lower:
                complexity_multiplier = max(0.5, complexity_multiplier - 0.3)
        
        # Adjust for step length (longer descriptions often indicate complexity)
        if len(step_text) > 100:
            complexity_multiplier += 0.3
        elif len(step_text) < 20:
            complexity_multiplier = max(0.3, complexity_multiplier - 0.2)
        
        return timedelta(seconds=int(base_duration.total_seconds() * complexity_multiplier))
    
    def _calculate_priority(self, step_text: str, step_index: int, task_type: TaskType) -> int:
        """Calculate priority for a sub-task."""
        # Base priority inversely related to step index (earlier steps are higher priority)
        base_priority = max(1, 10 - step_index)
        
        # Task type modifiers
        type_modifiers = {
            TaskType.ANALYSIS: 2,      # Analysis should happen early
            TaskType.COORDINATION: 1,  # Coordination is important
            TaskType.IMPLEMENTATION: 0,  # Neutral
            TaskType.TESTING: -1,      # Testing usually comes after implementation
            TaskType.DOCUMENTATION: -2,  # Documentation often comes last
            TaskType.VALIDATION: 1     # Validation is important
        }
        
        priority_modifier = type_modifiers.get(task_type, 0)
        
        # Keyword-based modifiers
        high_priority_keywords = ['setup', 'initialize', 'configure', 'prepare', 'analyze']
        low_priority_keywords = ['cleanup', 'document', 'finalize', 'archive']
        
        step_lower = step_text.lower()
        
        for keyword in high_priority_keywords:
            if keyword in step_lower:
                priority_modifier += 1
                break
        
        for keyword in low_priority_keywords:
            if keyword in step_lower:
                priority_modifier -= 1
                break
        
        return max(1, base_priority + priority_modifier)
    
    async def _should_further_decompose(self, subtask: SubTask, config: TaskDecomposerConfig) -> bool:
        """Determine if a subtask needs further decomposition."""
        # Check if description is complex enough to warrant decomposition
        description = subtask.description.lower()
        
        # Indicators that suggest further decomposition is needed
        decomposition_indicators = [
            'and also', 'then', 'after', 'next', 'followed by',
            'first', 'second', 'third', 'finally',
            'including', 'such as', 'for example'
        ]
        
        # Count indicators
        indicator_count = sum(1 for indicator in decomposition_indicators if indicator in description)
        
        # Long descriptions may need decomposition
        is_long = len(subtask.description) > 150
        
        # High estimated duration suggests complexity
        is_complex_duration = (
            subtask.estimated_duration and 
            subtask.estimated_duration > timedelta(hours=4)
        )
        
        return indicator_count >= 2 or (is_long and indicator_count >= 1) or is_complex_duration
    
    async def _decompose_subtask(self, subtask: SubTask, config: TaskDecomposerConfig) -> List[SubTask]:
        """Further decompose a complex subtask."""
        # Simple heuristic-based decomposition
        description = subtask.description
        
        # Try to split on common separators
        separators = ['. ', '; ', ' then ', ' and ', ' after ', ' next ']
        parts = [description]
        
        for separator in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(separator))
            parts = new_parts
        
        # Create sub-subtasks
        sub_subtasks = []
        for i, part in enumerate(parts):
            if len(part.strip()) > 10:  # Skip very short parts
                sub_subtask = SubTask(
                    name=f"{subtask.name} - Part {i+1}",
                    description=part.strip(),
                    task_type=subtask.task_type,
                    priority=subtask.priority,
                    estimated_duration=timedelta(
                        seconds=int(subtask.estimated_duration.total_seconds() / len(parts))
                    ) if subtask.estimated_duration else None,
                    required_resources=subtask.required_resources.copy(),
                    metadata={
                        **subtask.metadata,
                        'parent_task_id': subtask.id,
                        'decomposition_part': i
                    }
                )
                sub_subtasks.append(sub_subtask)
        
        return sub_subtasks if len(sub_subtasks) > 1 else [subtask]
    
    async def _add_implicit_tasks(
        self,
        subtasks: List[SubTask],
        approach: Approach,
        config: TaskDecomposerConfig
    ) -> List[SubTask]:
        """Add implicit tasks that may be needed."""
        enhanced_subtasks = subtasks.copy()
        
        # Check if setup task is needed
        has_setup = any('setup' in task.description.lower() or 'initialize' in task.description.lower() 
                       for task in subtasks)
        
        if not has_setup and len(subtasks) > 2:
            setup_task = SubTask(
                name="Setup and preparation",
                description="Initialize environment and prepare for implementation",
                task_type=TaskType.ANALYSIS,
                priority=10,  # High priority
                estimated_duration=timedelta(minutes=15),
                metadata={'implicit_task': True, 'task_category': 'setup'}
            )
            enhanced_subtasks.insert(0, setup_task)
        
        # Check if validation task is needed
        has_validation = any(task.task_type == TaskType.VALIDATION for task in subtasks)
        has_testing = any(task.task_type == TaskType.TESTING for task in subtasks)
        
        if not has_validation and not has_testing and len(subtasks) > 1:
            validation_task = SubTask(
                name="Final validation",
                description="Validate implementation meets requirements",
                task_type=TaskType.VALIDATION,
                priority=1,  # Low priority (done last)
                estimated_duration=timedelta(minutes=20),
                metadata={'implicit_task': True, 'task_category': 'validation'}
            )
            enhanced_subtasks.append(validation_task)
        
        return enhanced_subtasks
    
    async def _has_dependency(self, task_a: SubTask, task_b: SubTask) -> bool:
        """Check if task_b depends on task_a."""
        # Check explicit dependency patterns
        for pattern in self._dependency_patterns:
            if pattern['from_keywords']:
                task_a_matches = any(keyword in task_a.description.lower() 
                                   for keyword in pattern['from_keywords'])
            else:
                task_a_matches = True
                
            if pattern['to_keywords']:
                task_b_matches = any(keyword in task_b.description.lower() 
                                   for keyword in pattern['to_keywords'])
            else:
                task_b_matches = True
            
            if task_a_matches and task_b_matches:
                return True
        
        # Task type-based dependencies
        type_dependencies = {
            (TaskType.ANALYSIS, TaskType.IMPLEMENTATION): True,
            (TaskType.IMPLEMENTATION, TaskType.TESTING): True,
            (TaskType.TESTING, TaskType.VALIDATION): True,
            (TaskType.IMPLEMENTATION, TaskType.DOCUMENTATION): True,
        }
        
        return type_dependencies.get((task_a.task_type, task_b.task_type), False)
    
    async def _find_weakest_dependency_in_cycle(
        self,
        cycle: List[str],
        dependency_graph: DependencyGraph
    ) -> Optional[Tuple[str, str]]:
        """Find the weakest dependency link in a cycle to break."""
        if len(cycle) < 2:
            return None
        
        # Simple heuristic: break the link between tasks with the least type affinity
        weakest_link = None
        weakest_strength = float('inf')
        
        for i in range(len(cycle)):
            from_task_id = cycle[i]
            to_task_id = cycle[(i + 1) % len(cycle)]
            
            if from_task_id in dependency_graph.tasks and to_task_id in dependency_graph.tasks:
                from_task = dependency_graph.tasks[from_task_id]
                to_task = dependency_graph.tasks[to_task_id]
                
                # Calculate dependency strength based on task types and content
                strength = self._calculate_dependency_strength(from_task, to_task)
                
                if strength < weakest_strength:
                    weakest_strength = strength
                    weakest_link = (from_task_id, to_task_id)
        
        return weakest_link
    
    def _calculate_dependency_strength(self, from_task: SubTask, to_task: SubTask) -> float:
        """Calculate the strength of dependency between two tasks."""
        strength = 0.0
        
        # Task type affinity
        type_affinities = {
            (TaskType.ANALYSIS, TaskType.IMPLEMENTATION): 0.8,
            (TaskType.IMPLEMENTATION, TaskType.TESTING): 0.9,
            (TaskType.TESTING, TaskType.VALIDATION): 0.7,
            (TaskType.IMPLEMENTATION, TaskType.DOCUMENTATION): 0.5,
        }
        
        type_strength = type_affinities.get((from_task.task_type, to_task.task_type), 0.3)
        strength += type_strength
        
        # Content-based dependency indicators
        content_indicators = [
            ('output', 'input'), ('result', 'use'), ('create', 'deploy'),
            ('build', 'test'), ('implement', 'validate')
        ]
        
        from_desc = from_task.description.lower()
        to_desc = to_task.description.lower()
        
        for from_keyword, to_keyword in content_indicators:
            if from_keyword in from_desc and to_keyword in to_desc:
                strength += 0.3
        
        return strength
    
    async def _find_merge_candidates(self, dependency_graph: DependencyGraph) -> List[List[str]]:
        """Find tasks that could be merged for efficiency."""
        merge_candidates = []
        
        tasks = list(dependency_graph.tasks.values())
        
        for i, task_a in enumerate(tasks):
            for task_b in tasks[i+1:]:
                if await self._should_merge_tasks(task_a, task_b, dependency_graph):
                    # Check if either task is already in a merge group
                    existing_group = None
                    for group in merge_candidates:
                        if task_a.id in group or task_b.id in group:
                            existing_group = group
                            break
                    
                    if existing_group:
                        if task_a.id not in existing_group:
                            existing_group.append(task_a.id)
                        if task_b.id not in existing_group:
                            existing_group.append(task_b.id)
                    else:
                        merge_candidates.append([task_a.id, task_b.id])
        
        return merge_candidates
    
    async def _should_merge_tasks(self, task_a: SubTask, task_b: SubTask, dependency_graph: DependencyGraph) -> bool:
        """Determine if two tasks should be merged."""
        # Same task type and similar estimated duration
        if task_a.task_type != task_b.task_type:
            return False
        
        # Both should be small tasks
        if (task_a.estimated_duration and task_a.estimated_duration > timedelta(hours=1)) or \
           (task_b.estimated_duration and task_b.estimated_duration > timedelta(hours=1)):
            return False
        
        # No dependency between them
        if task_a.id in task_b.dependencies or task_b.id in task_a.dependencies:
            return False
        
        # Similar content or context
        desc_a = task_a.description.lower()
        desc_b = task_b.description.lower()
        
        # Simple word overlap check
        words_a = set(desc_a.split())
        words_b = set(desc_b.split())
        overlap = len(words_a & words_b) / len(words_a | words_b) if words_a | words_b else 0
        
        return overlap > 0.3
    
    async def _merge_tasks(self, task_ids: List[str], dependency_graph: DependencyGraph):
        """Merge a group of tasks into a single task."""
        if len(task_ids) < 2:
            return
        
        tasks = [dependency_graph.tasks[task_id] for task_id in task_ids]
        
        # Create merged task
        merged_task = SubTask(
            name=f"Merged task: {tasks[0].name}",
            description=f"Combined task: {'; '.join(task.description for task in tasks)}",
            task_type=tasks[0].task_type,
            priority=max(task.priority for task in tasks),
            estimated_duration=timedelta(
                seconds=sum(task.estimated_duration.total_seconds() if task.estimated_duration else 0 
                           for task in tasks)
            ),
            required_resources=list(set().union(*[task.required_resources for task in tasks])),
            metadata={'merged_from': task_ids, 'merged_task': True}
        )
        
        # Update dependency graph
        dependency_graph.add_task(merged_task)
        
        # Transfer dependencies
        for task_id in task_ids:
            # Remove old task
            if task_id in dependency_graph.tasks:
                del dependency_graph.tasks[task_id]
            
            # Update edges
            if task_id in dependency_graph.edges:
                for dependent_id in dependency_graph.edges[task_id]:
                    dependency_graph.add_dependency(dependent_id, merged_task.id)
                del dependency_graph.edges[task_id]
            
            # Update other tasks' dependencies
            for other_task in dependency_graph.tasks.values():
                if task_id in other_task.dependencies:
                    other_task.dependencies.remove(task_id)
                    if merged_task.id not in other_task.dependencies:
                        other_task.dependencies.append(merged_task.id)
    
    async def _find_split_candidates(self, dependency_graph: DependencyGraph) -> List[str]:
        """Find tasks that should be split for better parallelization."""
        split_candidates = []
        
        for task in dependency_graph.tasks.values():
            if await self._should_split_task(task, dependency_graph):
                split_candidates.append(task.id)
        
        return split_candidates
    
    async def _should_split_task(self, task: SubTask, dependency_graph: DependencyGraph) -> bool:
        """Determine if a task should be split."""
        # Large tasks with long duration
        if task.estimated_duration and task.estimated_duration > timedelta(hours=6):
            return True
        
        # Tasks with multiple distinct activities
        description = task.description.lower()
        activity_indicators = ['and', 'then', 'also', 'additionally', 'furthermore']
        
        indicator_count = sum(1 for indicator in activity_indicators if indicator in description)
        
        return indicator_count >= 3
    
    async def _split_task_for_parallelization(self, task_id: str, dependency_graph: DependencyGraph):
        """Split a task for better parallel execution."""
        if task_id not in dependency_graph.tasks:
            return
        
        task = dependency_graph.tasks[task_id]
        
        # Simple splitting based on conjunctions
        description = task.description
        split_points = []
        
        for conjunction in [' and ', ' then ', ' also ']:
            parts = description.split(conjunction)
            if len(parts) > 1:
                split_points = parts
                break
        
        if len(split_points) < 2:
            return
        
        # Create split tasks
        split_tasks = []
        for i, part in enumerate(split_points):
            if len(part.strip()) > 10:
                split_task = SubTask(
                    name=f"{task.name} - Part {i+1}",
                    description=part.strip(),
                    task_type=task.task_type,
                    priority=task.priority,
                    estimated_duration=timedelta(
                        seconds=int(task.estimated_duration.total_seconds() / len(split_points))
                    ) if task.estimated_duration else None,
                    required_resources=task.required_resources.copy(),
                    metadata={**task.metadata, 'split_from': task_id, 'split_part': i}
                )
                split_tasks.append(split_task)
        
        if len(split_tasks) > 1:
            # Add split tasks to graph
            for split_task in split_tasks:
                dependency_graph.add_task(split_task)
            
            # Transfer dependencies from original task
            if task_id in dependency_graph.edges:
                for dependent_id in dependency_graph.edges[task_id]:
                    # Last split task becomes the new dependency target
                    dependency_graph.add_dependency(dependent_id, split_tasks[-1].id)
                del dependency_graph.edges[task_id]
            
            # Remove original task
            del dependency_graph.tasks[task_id]
    
    def _assess_complexity(self, approach: Approach) -> int:
        """Assess the complexity of an approach."""
        complexity = 0
        
        # Number of steps
        complexity += len(approach.steps)
        
        # Length of description
        complexity += len(approach.description) // 100
        
        # Number of risks and benefits
        complexity += len(approach.risks) + len(approach.benefits)
        
        # Estimated effort
        if approach.estimated_effort:
            complexity += int(approach.estimated_effort)
        
        return complexity
    
    async def _create_single_task_from_approach(self, approach: Approach) -> SubTask:
        """Create a single task from a simple approach."""
        return SubTask(
            name=approach.name,
            description=approach.description,
            task_type=TaskType.IMPLEMENTATION,
            priority=5,
            estimated_duration=timedelta(hours=1),
            metadata={'single_task_approach': True, 'approach_id': approach.id}
        )
    
    def _build_task_type_patterns(self) -> Dict[TaskType, List[str]]:
        """Build patterns for classifying task types."""
        return {
            TaskType.ANALYSIS: ['analyze', 'investigate', 'research', 'examine', 'study', 'review', 'assess'],
            TaskType.IMPLEMENTATION: ['implement', 'create', 'build', 'develop', 'write', 'code', 'construct'],
            TaskType.TESTING: ['test', 'verify', 'validate', 'check', 'ensure', 'confirm', 'prove'],
            TaskType.DOCUMENTATION: ['document', 'write docs', 'explain', 'describe', 'record', 'note'],
            TaskType.COORDINATION: ['coordinate', 'sync', 'integrate', 'merge', 'combine', 'align'],
            TaskType.VALIDATION: ['review', 'approve', 'confirm', 'validate', 'accept', 'certify']
        }
    
    def _build_dependency_patterns(self) -> List[Dict[str, List[str]]]:
        """Build patterns for detecting dependencies."""
        return [
            {
                'from_keywords': ['setup', 'initialize', 'prepare'],
                'to_keywords': ['implement', 'create', 'build']
            },
            {
                'from_keywords': ['implement', 'create', 'build'],
                'to_keywords': ['test', 'verify', 'validate']
            },
            {
                'from_keywords': ['test', 'verify'],
                'to_keywords': ['deploy', 'release', 'publish']
            },
            {
                'from_keywords': ['analyze', 'research'],
                'to_keywords': ['design', 'plan', 'implement']
            },
            {
                'from_keywords': ['design', 'plan'],
                'to_keywords': ['implement', 'build', 'create']
            }
        ]
    
    def get_decomposition_stats(self) -> Dict[str, Any]:
        """Get decomposition performance statistics."""
        avg_subtasks = (
            self._total_subtasks_created / self._total_decompositions
            if self._total_decompositions > 0 else 0
        )
        
        return {
            "total_decompositions": self._total_decompositions,
            "total_subtasks_created": self._total_subtasks_created,
            "average_subtasks_per_decomposition": round(avg_subtasks, 2),
            "circular_dependencies_detected": self._circular_dependencies_detected,
            "config": {
                "max_depth": self.config.max_depth,
                "max_subtasks_total": self.config.max_subtasks_total,
                "enable_dependency_detection": self.config.enable_dependency_detection,
                "enable_parallel_task_identification": self.config.enable_parallel_task_identification
            }
        }