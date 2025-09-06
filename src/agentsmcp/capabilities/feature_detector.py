"""
Generic feature detection system for any codebase.
Identifies existing CLI features, capabilities, and functions before attempting implementation.
"""

import asyncio
import subprocess
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureDetectionResult:
    """Result of feature detection analysis."""
    exists: bool
    feature_type: str  # "cli_flag", "command", "function", "capability"
    detection_method: str
    evidence: List[str]
    usage_examples: List[str]
    related_features: List[str]
    confidence: float  # 0.0 to 1.0

class FeatureDetector:
    """Generic feature detector that works on any codebase."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.logger = logging.getLogger(__name__)
        
    async def detect_cli_feature(self, request: str) -> FeatureDetectionResult:
        """
        Detect if a CLI feature already exists based on user request.
        
        Args:
            request: User's feature request (e.g., "Add --version flag")
            
        Returns:
            FeatureDetectionResult with detection status and evidence
        """
        # Parse the request to extract feature intent
        feature_intent = self._parse_feature_intent(request)
        
        # FAST PATH: Only use source code analysis to avoid subprocess hangs
        # This dramatically improves performance while still catching most existing features
        source_result = await self._detect_via_source_analysis(feature_intent)
        if source_result.exists:
            return source_result
            
        # If source analysis finds nothing, return negative result quickly
        return FeatureDetectionResult(
            exists=False,
            feature_type="unknown", 
            detection_method="source_analysis_only",
            evidence=["Performed fast source code scan - no matching features found"],
            usage_examples=[],
            related_features=[],
            confidence=0.85  # Good confidence based on source analysis
        )
    
    def _parse_feature_intent(self, request: str) -> Dict[str, Any]:
        """Parse user request to extract feature intent."""
        request_lower = request.lower()
        
        # CLI flag patterns
        flag_patterns = [
            r"--(\w+)\s+flag",
            r"-(\w)\s+flag", 
            r"--(\w+)",
            r"-(\w)"
        ]
        
        intent = {
            "type": "unknown",
            "flags": [],
            "commands": [],
            "keywords": []
        }
        
        # Extract potential flags
        for pattern in flag_patterns:
            matches = re.findall(pattern, request_lower)
            intent["flags"].extend(matches)
        
        # More precise feature keyword detection - only match CLI implementation requests
        if any(phrase in request_lower for phrase in ["--version", "add version", "implement version", "create version", "version flag", "version option"]):
            intent["type"] = "version_info"
            intent["flags"].extend(["version", "v"])
            intent["keywords"].append("version")
            
        elif any(phrase in request_lower for phrase in ["--help", "add help", "implement help", "create help", "help flag", "help option"]):
            # Only trigger on explicit CLI help flag requests, not general help queries
            intent["type"] = "help_info"
            intent["flags"].extend(["help", "h"])
            intent["keywords"].append("help")
            
        elif any(phrase in request_lower for phrase in ["--debug", "add debug", "implement debug", "create debug", "debug flag", "debug option", "debug mode"]):
            intent["type"] = "debug_mode"
            intent["flags"].extend(["debug", "verbose", "d", "v"])
            intent["keywords"].append("debug")
            
        return intent
    
    async def _detect_via_cli_help(self, feature_intent: Dict) -> FeatureDetectionResult:
        """Detect features by analyzing CLI --help output."""
        try:
            # Find potential CLI executables
            cli_candidates = self._find_cli_executables()
            
            for cli_path in cli_candidates:
                try:
                    # Get help output using async subprocess with timeout
                    process = await asyncio.create_subprocess_exec(
                        str(cli_path), "--help",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    try:
                        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3.0)
                        result_stdout = stdout.decode('utf-8') if stdout else ""
                        result_returncode = process.returncode
                    except asyncio.TimeoutError:
                        # Kill the process if it times out
                        try:
                            process.kill()
                            await process.wait()
                        except:
                            pass
                        continue
                    
                    if result_returncode == 0:
                        help_text = result_stdout.lower()
                        
                        # Check for flags
                        evidence = []
                        usage_examples = []
                        
                        for flag in feature_intent.get("flags", []):
                            if f"--{flag}" in help_text or f"-{flag}" in help_text:
                                evidence.append(f"Found --{flag} in help output")
                                usage_examples.append(f"{cli_path.name} --{flag}")
                        
                        if evidence:
                            return FeatureDetectionResult(
                                exists=True,
                                feature_type="cli_flag",
                                detection_method="help_analysis",
                                evidence=evidence,
                                usage_examples=usage_examples,
                                related_features=self._extract_related_features(help_text),
                                confidence=0.95
                            )
                            
                except (OSError, asyncio.TimeoutError, Exception):
                    continue
                    
        except Exception as e:
            self.logger.debug(f"CLI help detection failed: {e}")
            
        return FeatureDetectionResult(
            exists=False,
            feature_type="cli_flag", 
            detection_method="help_analysis",
            evidence=[],
            usage_examples=[],
            related_features=[],
            confidence=0.8
        )
    
    async def _detect_via_direct_test(self, feature_intent: Dict) -> FeatureDetectionResult:
        """Test features by running commands directly."""
        try:
            cli_candidates = self._find_cli_executables()
            
            for cli_path in cli_candidates:
                for flag in feature_intent.get("flags", []):
                    try:
                        # Test the flag directly using async subprocess
                        process = await asyncio.create_subprocess_exec(
                            str(cli_path), f"--{flag}",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        
                        try:
                            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=2.0)
                            result_stdout = stdout.decode('utf-8') if stdout else ""
                            result_returncode = process.returncode
                        except asyncio.TimeoutError:
                            # Kill the process if it times out
                            try:
                                process.kill()
                                await process.wait()
                            except:
                                pass
                            continue
                        
                        # If command ran successfully and produced output
                        if result_returncode == 0 and result_stdout.strip():
                            return FeatureDetectionResult(
                                exists=True,
                                feature_type="cli_flag",
                                detection_method="direct_execution",
                                evidence=[f"Command '{cli_path.name} --{flag}' executed successfully"],
                                usage_examples=[f"{cli_path.name} --{flag}"],
                                related_features=[],
                                confidence=1.0  # Highest confidence - we saw it work
                            )
                            
                    except (OSError, asyncio.TimeoutError, Exception):
                        continue
                        
        except Exception as e:
            self.logger.debug(f"Direct test detection failed: {e}")
            
        return FeatureDetectionResult(
            exists=False,
            feature_type="cli_flag",
            detection_method="direct_execution", 
            evidence=[],
            usage_examples=[],
            related_features=[],
            confidence=0.7
        )
    
    async def _detect_via_source_analysis(self, feature_intent: Dict) -> FeatureDetectionResult:
        """Detect features by analyzing source code."""
        try:
            # Find main CLI files
            cli_files = self._find_cli_source_files()
            
            evidence = []
            for cli_file in cli_files:
                try:
                    with open(cli_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for flag in feature_intent.get("flags", []):
                        # Look for flag definitions in various formats
                        patterns = [
                            rf'--{flag}["\'\s]',
                            rf'"-{flag}["\'\s]',  
                            rf'add_argument.*--{flag}',
                            rf'@click\.option.*--{flag}',
                            rf'version.*=.*{flag}'
                        ]
                        
                        for pattern in patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                evidence.append(f"Found --{flag} definition in {cli_file.name}")
                                
                except (IOError, UnicodeDecodeError):
                    continue
                    
            if evidence:
                return FeatureDetectionResult(
                    exists=True,
                    feature_type="cli_flag",
                    detection_method="source_analysis",
                    evidence=evidence,
                    usage_examples=[],
                    related_features=[],
                    confidence=0.8
                )
                
        except Exception as e:
            self.logger.debug(f"Source analysis detection failed: {e}")
            
        return FeatureDetectionResult(
            exists=False,
            feature_type="cli_flag",
            detection_method="source_analysis",
            evidence=[],
            usage_examples=[], 
            related_features=[],
            confidence=0.6
        )
    
    def _find_cli_executables(self) -> List[Path]:
        """Find potential CLI executables in the project."""
        candidates = []
        
        # Look for executables in common locations
        patterns = [
            self.project_root / "*",  # Root level executables
            self.project_root / "bin" / "*",
            self.project_root / "scripts" / "*"
        ]
        
        for pattern in patterns:
            for path in pattern.parent.glob(pattern.name):
                if path.is_file() and os.access(path, os.X_OK):
                    # Skip common non-CLI files
                    if path.suffix not in ['.txt', '.md', '.json', '.yaml', '.yml']:
                        candidates.append(path)
                        
        return candidates
    
    def _find_cli_source_files(self) -> List[Path]:
        """Find source files likely to contain CLI definitions."""
        candidates = []
        
        # Common CLI source file patterns
        patterns = [
            "*/cli.py",
            "*/main.py", 
            "*/__main__.py",
            "*/cli/*",
            "*/commands/*"
        ]
        
        for pattern in patterns:
            for path in self.project_root.glob(pattern):
                if path.is_file() and path.suffix == '.py':
                    candidates.append(path)
                    
        return candidates
    
    def _extract_related_features(self, help_text: str) -> List[str]:
        """Extract related features from help text."""
        related = []
        
        # Look for other flags in the help
        flag_pattern = r'--(\w+)'
        flags = re.findall(flag_pattern, help_text)
        
        # Return up to 3 related flags
        return flags[:3]
    
    async def generate_feature_showcase(self, result: FeatureDetectionResult) -> str:
        """Generate a formatted showcase when feature already exists."""
        if not result.exists:
            return ""
            
        # Build formatted showcase
        lines = []
        lines.append("🎯 **Feature Already Available**")
        lines.append("")
        lines.append(f"✅ This {result.feature_type} already exists in the codebase")
        lines.append("")
        
        if result.usage_examples:
            lines.append("**Try it now:**")
            for example in result.usage_examples:
                lines.append(f"```bash")
                lines.append(f"$ {example}")
                lines.append("```")
                lines.append("")
        
        if result.related_features:
            lines.append("**Related features you might like:**")
            for feature in result.related_features:
                lines.append(f"• `--{feature}`")
            lines.append("")
            
        if result.evidence:
            lines.append("**Detection evidence:**")
            for evidence in result.evidence:
                lines.append(f"• {evidence}")
        
        return "\n".join(lines)