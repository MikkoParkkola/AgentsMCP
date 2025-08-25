"""
Multi-Modal Agent Capabilities Engine

Extends AgentsMCP to handle diverse content modalities beyond text processing,
including code analysis, image processing, data manipulation, and structured outputs.
"""

import asyncio
import hashlib
import json
import mimetypes
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported content modalities"""
    TEXT = "text"
    CODE = "code"  
    IMAGE = "image"
    DATA = "data"
    STRUCTURED = "structured"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"


class ProcessingCapability(Enum):
    """Agent processing capabilities"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    SYNTHESIS = "synthesis"
    EXTRACTION = "extraction"


@dataclass
class ModalContent:
    """Container for multi-modal content"""
    content_id: str
    modality: ModalityType
    data: Union[str, bytes, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    mime_type: Optional[str] = None
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.data, (str, bytes)):
            self.size_bytes = len(self.data) if isinstance(self.data, str) else len(self.data)
        elif isinstance(self.data, dict):
            self.size_bytes = len(json.dumps(self.data).encode('utf-8'))


@dataclass  
class AgentCapabilityProfile:
    """Defines an agent's multi-modal processing capabilities"""
    agent_id: str
    supported_modalities: Set[ModalityType]
    capabilities: Set[ProcessingCapability]
    max_content_size: Dict[ModalityType, int] = field(default_factory=dict)
    processing_speed: Dict[ModalityType, float] = field(default_factory=dict)  # items/sec
    quality_scores: Dict[Tuple[ModalityType, ProcessingCapability], float] = field(default_factory=dict)
    cost_per_operation: Dict[ModalityType, float] = field(default_factory=dict)
    specializations: Set[str] = field(default_factory=set)  # e.g., "python", "typescript", "cv", "nlp"


class ModalProcessor(ABC):
    """Abstract base class for modality-specific processors"""
    
    @abstractmethod
    async def can_process(self, content: ModalContent, capability: ProcessingCapability) -> bool:
        pass
    
    @abstractmethod
    async def process(self, content: ModalContent, capability: ProcessingCapability, 
                     parameters: Dict[str, Any] = None) -> ModalContent:
        pass
    
    @abstractmethod
    async def estimate_cost(self, content: ModalContent, capability: ProcessingCapability) -> float:
        pass
    
    @abstractmethod
    async def estimate_duration(self, content: ModalContent, capability: ProcessingCapability) -> float:
        pass


class CodeProcessor(ModalProcessor):
    """Specialized processor for code content"""
    
    def __init__(self):
        self.supported_languages = {
            "python", "javascript", "typescript", "java", "c++", "c", "go", 
            "rust", "scala", "kotlin", "swift", "ruby", "php", "sql", "html", "css"
        }
        
    async def can_process(self, content: ModalContent, capability: ProcessingCapability) -> bool:
        if content.modality != ModalityType.CODE:
            return False
            
        language = content.metadata.get("language", "").lower()
        return language in self.supported_languages
    
    async def process(self, content: ModalContent, capability: ProcessingCapability,
                     parameters: Dict[str, Any] = None) -> ModalContent:
        parameters = parameters or {}
        
        if capability == ProcessingCapability.ANALYSIS:
            return await self._analyze_code(content, parameters)
        elif capability == ProcessingCapability.OPTIMIZATION:
            return await self._optimize_code(content, parameters)
        elif capability == ProcessingCapability.VALIDATION:
            return await self._validate_code(content, parameters)
        elif capability == ProcessingCapability.TRANSFORMATION:
            return await self._transform_code(content, parameters)
        else:
            raise ValueError(f"Unsupported capability: {capability}")
    
    async def _analyze_code(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Analyze code structure, complexity, and quality metrics"""
        code_text = content.data if isinstance(content.data, str) else str(content.data)
        
        # Simulate code analysis
        analysis_results = {
            "lines_of_code": len(code_text.splitlines()),
            "complexity_estimate": min(10, len(code_text) // 100),
            "functions_detected": code_text.count("def ") + code_text.count("function "),
            "imports_detected": code_text.count("import ") + code_text.count("#include"),
            "language": content.metadata.get("language", "unknown"),
            "quality_score": 8.5,  # Mock quality score
            "security_issues": [],  # Would integrate with actual security scanners
            "performance_suggestions": []
        }
        
        return ModalContent(
            content_id=f"analysis_{content.content_id}",
            modality=ModalityType.STRUCTURED,
            data=analysis_results,
            metadata={"source_content_id": content.content_id, "analysis_type": "code_analysis"}
        )
    
    async def _optimize_code(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Optimize code for performance and readability"""
        # Mock optimization - would integrate with actual code optimization tools
        optimized_code = content.data  # Placeholder
        
        return ModalContent(
            content_id=f"optimized_{content.content_id}",
            modality=ModalityType.CODE,
            data=optimized_code,
            metadata={**content.metadata, "optimized": True, "optimization_level": parameters.get("level", "standard")}
        )
    
    async def _validate_code(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Validate code syntax and semantics"""
        validation_results = {
            "syntax_valid": True,  # Mock validation
            "semantic_issues": [],
            "warnings": [],
            "suggestions": []
        }
        
        return ModalContent(
            content_id=f"validation_{content.content_id}",
            modality=ModalityType.STRUCTURED,
            data=validation_results,
            metadata={"source_content_id": content.content_id, "validation_type": "code_validation"}
        )
    
    async def _transform_code(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Transform code between languages or styles"""
        target_language = parameters.get("target_language", "python")
        
        # Mock transformation
        transformed_code = f"# Transformed to {target_language}\n{content.data}"
        
        return ModalContent(
            content_id=f"transformed_{content.content_id}",
            modality=ModalityType.CODE,
            data=transformed_code,
            metadata={**content.metadata, "language": target_language, "transformed": True}
        )
    
    async def estimate_cost(self, content: ModalContent, capability: ProcessingCapability) -> float:
        base_cost = 0.01
        size_multiplier = content.size_bytes / 1000  # Cost per KB
        complexity_multiplier = {
            ProcessingCapability.ANALYSIS: 1.0,
            ProcessingCapability.OPTIMIZATION: 2.0,
            ProcessingCapability.VALIDATION: 0.5,
            ProcessingCapability.TRANSFORMATION: 1.5
        }.get(capability, 1.0)
        
        return base_cost * size_multiplier * complexity_multiplier
    
    async def estimate_duration(self, content: ModalContent, capability: ProcessingCapability) -> float:
        base_duration = 1.0  # seconds
        size_factor = content.size_bytes / 10000  # Duration per 10KB
        complexity_factor = {
            ProcessingCapability.ANALYSIS: 1.0,
            ProcessingCapability.OPTIMIZATION: 3.0,
            ProcessingCapability.VALIDATION: 0.5,
            ProcessingCapability.TRANSFORMATION: 2.0
        }.get(capability, 1.0)
        
        return base_duration + (size_factor * complexity_factor)


class DataProcessor(ModalProcessor):
    """Specialized processor for structured data content"""
    
    def __init__(self):
        self.supported_formats = {"json", "csv", "xml", "yaml", "parquet", "avro"}
    
    async def can_process(self, content: ModalContent, capability: ProcessingCapability) -> bool:
        if content.modality != ModalityType.DATA:
            return False
        
        data_format = content.metadata.get("format", "").lower()
        return data_format in self.supported_formats
    
    async def process(self, content: ModalContent, capability: ProcessingCapability,
                     parameters: Dict[str, Any] = None) -> ModalContent:
        parameters = parameters or {}
        
        if capability == ProcessingCapability.ANALYSIS:
            return await self._analyze_data(content, parameters)
        elif capability == ProcessingCapability.TRANSFORMATION:
            return await self._transform_data(content, parameters)
        elif capability == ProcessingCapability.VALIDATION:
            return await self._validate_data(content, parameters)
        elif capability == ProcessingCapability.EXTRACTION:
            return await self._extract_insights(content, parameters)
        else:
            raise ValueError(f"Unsupported capability: {capability}")
    
    async def _analyze_data(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Analyze data structure, quality, and characteristics"""
        if isinstance(content.data, dict):
            data = content.data
        else:
            try:
                data = json.loads(content.data) if isinstance(content.data, str) else content.data
            except:
                data = {}
        
        analysis = {
            "record_count": len(data) if isinstance(data, list) else 1,
            "schema": self._infer_schema(data),
            "data_quality": self._assess_data_quality(data),
            "statistics": self._compute_statistics(data),
            "completeness": self._assess_completeness(data)
        }
        
        return ModalContent(
            content_id=f"analysis_{content.content_id}",
            modality=ModalityType.STRUCTURED,
            data=analysis,
            metadata={"source_content_id": content.content_id, "analysis_type": "data_analysis"}
        )
    
    def _infer_schema(self, data: Any) -> Dict[str, Any]:
        """Infer schema from data structure"""
        if isinstance(data, dict):
            return {key: type(value).__name__ for key, value in data.items()}
        elif isinstance(data, list) and data:
            return {"array_type": type(data[0]).__name__, "length": len(data)}
        else:
            return {"type": type(data).__name__}
    
    def _assess_data_quality(self, data: Any) -> Dict[str, Any]:
        """Assess data quality metrics"""
        return {
            "quality_score": 8.7,  # Mock score
            "issues": [],
            "suggestions": ["Consider data normalization", "Add missing value handling"]
        }
    
    def _compute_statistics(self, data: Any) -> Dict[str, Any]:
        """Compute basic statistics"""
        return {
            "size": len(str(data)),
            "nested_levels": self._count_nesting_levels(data),
            "unique_keys": len(set(data.keys())) if isinstance(data, dict) else 0
        }
    
    def _count_nesting_levels(self, obj: Any, level: int = 0) -> int:
        """Count maximum nesting levels in data structure"""
        if isinstance(obj, dict):
            return max([self._count_nesting_levels(v, level + 1) for v in obj.values()] + [level])
        elif isinstance(obj, list):
            return max([self._count_nesting_levels(item, level + 1) for item in obj] + [level])
        else:
            return level
    
    def _assess_completeness(self, data: Any) -> Dict[str, Any]:
        """Assess data completeness"""
        return {
            "completeness_score": 0.95,  # Mock score
            "missing_fields": [],
            "null_counts": {}
        }
    
    async def _transform_data(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Transform data format or structure"""
        target_format = parameters.get("target_format", "json")
        transformed_data = content.data  # Mock transformation
        
        return ModalContent(
            content_id=f"transformed_{content.content_id}",
            modality=ModalityType.DATA,
            data=transformed_data,
            metadata={**content.metadata, "format": target_format, "transformed": True}
        )
    
    async def _validate_data(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Validate data against schema or rules"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        return ModalContent(
            content_id=f"validation_{content.content_id}",
            modality=ModalityType.STRUCTURED,
            data=validation_results,
            metadata={"source_content_id": content.content_id, "validation_type": "data_validation"}
        )
    
    async def _extract_insights(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Extract insights and patterns from data"""
        insights = {
            "patterns": ["Time-series trend detected", "Seasonal variation present"],
            "anomalies": [],
            "correlations": {},
            "recommendations": ["Consider data aggregation", "Apply time-series analysis"]
        }
        
        return ModalContent(
            content_id=f"insights_{content.content_id}",
            modality=ModalityType.STRUCTURED,
            data=insights,
            metadata={"source_content_id": content.content_id, "analysis_type": "insight_extraction"}
        )
    
    async def estimate_cost(self, content: ModalContent, capability: ProcessingCapability) -> float:
        base_cost = 0.005
        size_multiplier = content.size_bytes / 1000
        complexity_multiplier = {
            ProcessingCapability.ANALYSIS: 1.0,
            ProcessingCapability.TRANSFORMATION: 1.2,
            ProcessingCapability.VALIDATION: 0.3,
            ProcessingCapability.EXTRACTION: 1.5
        }.get(capability, 1.0)
        
        return base_cost * size_multiplier * complexity_multiplier
    
    async def estimate_duration(self, content: ModalContent, capability: ProcessingCapability) -> float:
        base_duration = 0.5
        size_factor = content.size_bytes / 5000
        return base_duration + size_factor


class ImageProcessor(ModalProcessor):
    """Specialized processor for image content"""
    
    def __init__(self):
        self.supported_formats = {"jpeg", "jpg", "png", "gif", "bmp", "tiff", "webp"}
    
    async def can_process(self, content: ModalContent, capability: ProcessingCapability) -> bool:
        if content.modality != ModalityType.IMAGE:
            return False
        
        mime_type = content.mime_type or ""
        return any(fmt in mime_type.lower() for fmt in self.supported_formats)
    
    async def process(self, content: ModalContent, capability: ProcessingCapability,
                     parameters: Dict[str, Any] = None) -> ModalContent:
        parameters = parameters or {}
        
        if capability == ProcessingCapability.ANALYSIS:
            return await self._analyze_image(content, parameters)
        elif capability == ProcessingCapability.TRANSFORMATION:
            return await self._transform_image(content, parameters)
        elif capability == ProcessingCapability.EXTRACTION:
            return await self._extract_features(content, parameters)
        else:
            raise ValueError(f"Unsupported capability: {capability}")
    
    async def _analyze_image(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Analyze image properties and content"""
        analysis = {
            "dimensions": content.metadata.get("dimensions", {"width": 1920, "height": 1080}),
            "file_size": content.size_bytes,
            "format": content.mime_type,
            "color_analysis": {
                "dominant_colors": ["#3366CC", "#FFFFFF", "#000000"],
                "color_palette": 256,
                "brightness": 0.7,
                "contrast": 0.8
            },
            "content_detection": {
                "objects": ["person", "car", "building"],
                "text_detected": True,
                "faces_count": 2,
                "scene_type": "outdoor"
            },
            "technical_quality": {
                "sharpness": 0.85,
                "noise_level": 0.15,
                "exposure": 0.9
            }
        }
        
        return ModalContent(
            content_id=f"analysis_{content.content_id}",
            modality=ModalityType.STRUCTURED,
            data=analysis,
            metadata={"source_content_id": content.content_id, "analysis_type": "image_analysis"}
        )
    
    async def _transform_image(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Transform image (resize, format conversion, etc.)"""
        # Mock transformation
        transformed_data = content.data  # Would use actual image processing library
        
        return ModalContent(
            content_id=f"transformed_{content.content_id}",
            modality=ModalityType.IMAGE,
            data=transformed_data,
            metadata={**content.metadata, "transformed": True, **parameters}
        )
    
    async def _extract_features(self, content: ModalContent, parameters: Dict[str, Any]) -> ModalContent:
        """Extract features from image"""
        features = {
            "feature_vectors": [0.1, 0.3, 0.7, 0.9],  # Mock feature vector
            "keypoints": 142,
            "descriptors": "SIFT",
            "histogram": [25, 45, 67, 89, 12],  # Mock histogram
            "texture_features": {"contrast": 0.8, "homogeneity": 0.6, "energy": 0.7}
        }
        
        return ModalContent(
            content_id=f"features_{content.content_id}",
            modality=ModalityType.STRUCTURED,
            data=features,
            metadata={"source_content_id": content.content_id, "extraction_type": "image_features"}
        )
    
    async def estimate_cost(self, content: ModalContent, capability: ProcessingCapability) -> float:
        base_cost = 0.02
        size_multiplier = content.size_bytes / 100000  # Cost per 100KB
        complexity_multiplier = {
            ProcessingCapability.ANALYSIS: 1.0,
            ProcessingCapability.TRANSFORMATION: 0.8,
            ProcessingCapability.EXTRACTION: 1.5
        }.get(capability, 1.0)
        
        return base_cost * size_multiplier * complexity_multiplier
    
    async def estimate_duration(self, content: ModalContent, capability: ProcessingCapability) -> float:
        base_duration = 2.0
        size_factor = content.size_bytes / 50000
        return base_duration + size_factor


class MultiModalEngine:
    """Main engine for coordinating multi-modal agent capabilities"""
    
    def __init__(self, max_concurrent_tasks: int = 10, cache_size_mb: int = 100):
        self.agent_profiles: Dict[str, AgentCapabilityProfile] = {}
        self.processors: Dict[ModalityType, ModalProcessor] = {
            ModalityType.CODE: CodeProcessor(),
            ModalityType.DATA: DataProcessor(), 
            ModalityType.IMAGE: ImageProcessor()
        }
        self.content_cache: Dict[str, ModalContent] = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        self.cache_size_mb = cache_size_mb
        self.active_tasks: Set[str] = set()
        self.processing_history: List[Dict[str, Any]] = []
        
    async def register_agent(self, profile: AgentCapabilityProfile):
        """Register an agent's multi-modal capabilities"""
        self.agent_profiles[profile.agent_id] = profile
        logger.info(f"Registered agent {profile.agent_id} with capabilities: {profile.capabilities}")
    
    async def store_content(self, content: ModalContent) -> str:
        """Store content in the multi-modal cache"""
        self.content_cache[content.content_id] = content
        await self._manage_cache_size()
        return content.content_id
    
    async def retrieve_content(self, content_id: str) -> Optional[ModalContent]:
        """Retrieve content from the cache"""
        return self.content_cache.get(content_id)
    
    async def find_capable_agents(self, modality: ModalityType, capability: ProcessingCapability,
                                 content_size: int = 0, specialization: Optional[str] = None) -> List[str]:
        """Find agents capable of processing specific modality and capability"""
        capable_agents = []
        
        for agent_id, profile in self.agent_profiles.items():
            # Check modality support
            if modality not in profile.supported_modalities:
                continue
                
            # Check capability support  
            if capability not in profile.capabilities:
                continue
                
            # Check size limits
            max_size = profile.max_content_size.get(modality, float('inf'))
            if content_size > max_size:
                continue
                
            # Check specialization if required
            if specialization and specialization not in profile.specializations:
                continue
                
            capable_agents.append(agent_id)
        
        # Sort by quality score if available
        def get_quality_score(agent_id: str) -> float:
            profile = self.agent_profiles[agent_id]
            return profile.quality_scores.get((modality, capability), 0.5)
        
        capable_agents.sort(key=get_quality_score, reverse=True)
        return capable_agents
    
    async def process_content(self, content_id: str, capability: ProcessingCapability,
                             agent_id: Optional[str] = None, parameters: Dict[str, Any] = None) -> ModalContent:
        """Process content using specified capability"""
        content = await self.retrieve_content(content_id)
        if not content:
            raise ValueError(f"Content not found: {content_id}")
        
        # Find appropriate agent if not specified
        if not agent_id:
            capable_agents = await self.find_capable_agents(
                content.modality, capability, content.size_bytes
            )
            if not capable_agents:
                raise ValueError(f"No agents capable of {capability} for {content.modality}")
            agent_id = capable_agents[0]
        
        # Check if agent is actually capable
        profile = self.agent_profiles.get(agent_id)
        if not profile or content.modality not in profile.supported_modalities:
            raise ValueError(f"Agent {agent_id} cannot process {content.modality}")
        
        # Get appropriate processor
        processor = self.processors.get(content.modality)
        if not processor:
            raise ValueError(f"No processor available for {content.modality}")
        
        # Estimate cost and duration
        estimated_cost = await processor.estimate_cost(content, capability)
        estimated_duration = await processor.estimate_duration(content, capability)
        
        # Create task ID and track
        task_id = f"{content_id}_{capability.value}_{agent_id}"
        self.active_tasks.add(task_id)
        
        try:
            # Process content
            start_time = datetime.now()
            result = await processor.process(content, capability, parameters)
            end_time = datetime.now()
            
            # Record processing history
            self.processing_history.append({
                "task_id": task_id,
                "content_id": content_id,
                "agent_id": agent_id,
                "modality": content.modality.value,
                "capability": capability.value,
                "estimated_cost": estimated_cost,
                "estimated_duration": estimated_duration,
                "actual_duration": (end_time - start_time).total_seconds(),
                "result_content_id": result.content_id,
                "timestamp": start_time.isoformat()
            })
            
            # Store result
            await self.store_content(result)
            
            return result
            
        finally:
            self.active_tasks.discard(task_id)
    
    async def batch_process(self, content_ids: List[str], capability: ProcessingCapability,
                           parameters: Dict[str, Any] = None) -> List[ModalContent]:
        """Process multiple pieces of content in parallel"""
        if len(content_ids) > self.max_concurrent_tasks:
            # Process in chunks
            results = []
            for i in range(0, len(content_ids), self.max_concurrent_tasks):
                chunk = content_ids[i:i + self.max_concurrent_tasks]
                chunk_results = await asyncio.gather(*[
                    self.process_content(cid, capability, parameters=parameters) 
                    for cid in chunk
                ])
                results.extend(chunk_results)
            return results
        else:
            # Process all in parallel
            return await asyncio.gather(*[
                self.process_content(cid, capability, parameters=parameters)
                for cid in content_ids
            ])
    
    async def create_content_pipeline(self, initial_content_id: str, 
                                    pipeline: List[Tuple[ProcessingCapability, Optional[str], Optional[Dict[str, Any]]]]) -> List[ModalContent]:
        """Execute a pipeline of processing steps on content"""
        current_content_id = initial_content_id
        results = []
        
        for capability, agent_id, parameters in pipeline:
            result = await self.process_content(current_content_id, capability, agent_id, parameters)
            results.append(result)
            current_content_id = result.content_id
        
        return results
    
    async def get_capability_matrix(self) -> Dict[str, Dict[str, List[str]]]:
        """Get a matrix of which agents support which modality-capability combinations"""
        matrix = {}
        
        for modality in ModalityType:
            matrix[modality.value] = {}
            for capability in ProcessingCapability:
                capable_agents = await self.find_capable_agents(modality, capability)
                matrix[modality.value][capability.value] = capable_agents
        
        return matrix
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about multi-modal processing"""
        total_tasks = len(self.processing_history)
        if total_tasks == 0:
            return {"total_tasks": 0}
        
        modality_counts = {}
        capability_counts = {}
        agent_usage = {}
        total_duration = 0
        total_cost = 0
        
        for task in self.processing_history:
            modality_counts[task["modality"]] = modality_counts.get(task["modality"], 0) + 1
            capability_counts[task["capability"]] = capability_counts.get(task["capability"], 0) + 1
            agent_usage[task["agent_id"]] = agent_usage.get(task["agent_id"], 0) + 1
            total_duration += task["actual_duration"]
            total_cost += task["estimated_cost"]
        
        return {
            "total_tasks": total_tasks,
            "active_tasks": len(self.active_tasks),
            "cached_content_items": len(self.content_cache),
            "modality_distribution": modality_counts,
            "capability_distribution": capability_counts,
            "agent_usage": agent_usage,
            "average_duration": total_duration / total_tasks,
            "total_estimated_cost": total_cost,
            "cache_utilization": len(self.content_cache)
        }
    
    async def optimize_agent_assignments(self) -> Dict[str, Any]:
        """Analyze and optimize agent task assignments based on performance history"""
        if len(self.processing_history) < 10:
            return {"message": "Insufficient data for optimization"}
        
        # Analyze agent performance by modality and capability
        agent_performance = {}
        
        for task in self.processing_history:
            key = (task["agent_id"], task["modality"], task["capability"])
            if key not in agent_performance:
                agent_performance[key] = {"durations": [], "costs": []}
            
            agent_performance[key]["durations"].append(task["actual_duration"])
            agent_performance[key]["costs"].append(task["estimated_cost"])
        
        # Find optimal assignments
        recommendations = {}
        
        for (agent_id, modality, capability), perf_data in agent_performance.items():
            avg_duration = sum(perf_data["durations"]) / len(perf_data["durations"])
            avg_cost = sum(perf_data["costs"]) / len(perf_data["costs"])
            
            combo_key = f"{modality}_{capability}"
            if combo_key not in recommendations:
                recommendations[combo_key] = {"best_agent": agent_id, "avg_duration": avg_duration, "avg_cost": avg_cost}
            else:
                current_best = recommendations[combo_key]
                # Prefer agents with better duration and cost balance
                if (avg_duration + avg_cost) < (current_best["avg_duration"] + current_best["avg_cost"]):
                    recommendations[combo_key] = {"best_agent": agent_id, "avg_duration": avg_duration, "avg_cost": avg_cost}
        
        return {
            "agent_performance_analysis": agent_performance,
            "recommended_assignments": recommendations,
            "optimization_basis": "duration_and_cost_balance"
        }
    
    async def _manage_cache_size(self):
        """Manage cache size by removing old or large content"""
        if len(self.content_cache) == 0:
            return
        
        # Calculate current cache size
        current_size_mb = sum(content.size_bytes for content in self.content_cache.values()) / (1024 * 1024)
        
        if current_size_mb > self.cache_size_mb:
            # Remove oldest items first
            sorted_content = sorted(
                self.content_cache.items(),
                key=lambda x: x[1].created_at
            )
            
            while current_size_mb > self.cache_size_mb * 0.8 and sorted_content:  # Target 80% of limit
                content_id, content = sorted_content.pop(0)
                del self.content_cache[content_id]
                current_size_mb -= content.size_bytes / (1024 * 1024)
                
                logger.info(f"Removed content {content_id} from cache due to size limits")