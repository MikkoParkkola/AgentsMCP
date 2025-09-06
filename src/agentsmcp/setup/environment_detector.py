"""Environment Detection System for AgentsMCP Installation.

Automatically detects:
- Operating system (Windows, macOS, Linux distributions)
- Python version and virtual environment status
- Shell type and terminal capabilities  
- Existing MCP servers and development tools
- Project context (Git repo, package.json, requirements.txt, etc.)
- Hardware capabilities and system resources
"""

from __future__ import annotations

import os
import sys
import platform
import shutil
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information detected during environment scan."""
    os_name: str
    os_version: str
    architecture: str
    python_version: str
    python_executable: str
    in_venv: bool
    venv_path: Optional[str] = None
    shell: str = "unknown"
    terminal: str = "unknown"
    supports_colors: bool = False
    supports_unicode: bool = False
    cpu_count: int = 1
    memory_gb: float = 0.0


@dataclass
class DevelopmentTools:
    """Development tools and services detected."""
    git_available: bool = False
    docker_available: bool = False
    node_available: bool = False
    npm_available: bool = False
    pip_available: bool = False
    conda_available: bool = False
    ollama_available: bool = False
    ollama_running: bool = False
    ollama_models: List[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    """Context about the current project/directory."""
    is_git_repo: bool = False
    has_requirements_txt: bool = False
    has_setup_py: bool = False
    has_pyproject_toml: bool = False
    has_package_json: bool = False
    has_dockerfile: bool = False
    has_docker_compose: bool = False
    detected_project_type: str = "unknown"
    

@dataclass
class APIKeysDetection:
    """Detected API keys and credentials."""
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None
    google_ai_key: Optional[str] = None
    openrouter_key: Optional[str] = None
    ollama_key: Optional[str] = None
    detected_providers: Set[str] = field(default_factory=set)


@dataclass
class MCPServers:
    """Detected MCP servers and configurations."""
    config_files: List[Path] = field(default_factory=list)
    running_servers: List[Dict[str, Any]] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)


@dataclass
class EnvironmentProfile:
    """Complete environment profile."""
    system: SystemInfo
    tools: DevelopmentTools
    project: ProjectContext
    api_keys: APIKeysDetection
    mcp_servers: MCPServers
    installation_mode: str = "auto"
    detected_at: float = 0.0
    

class EnvironmentDetector:
    """Intelligent environment detection system."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize environment detector.
        
        Args:
            verbose: Enable verbose logging during detection
        """
        self.verbose = verbose
        self._cache: Optional[EnvironmentProfile] = None
        
    def detect_environment(self, force_refresh: bool = False) -> EnvironmentProfile:
        """
        Perform comprehensive environment detection.
        
        Args:
            force_refresh: Force a new detection even if cached results exist
            
        Returns:
            Complete environment profile with all detected information
        """
        if self._cache is not None and not force_refresh:
            return self._cache
            
        if self.verbose:
            print("🔍 Analyzing environment...")
            
        profile = EnvironmentProfile(
            system=self._detect_system_info(),
            tools=self._detect_development_tools(),
            project=self._detect_project_context(),
            api_keys=self._detect_api_keys(),
            mcp_servers=self._detect_mcp_servers(),
            detected_at=time.time()
        )
        
        # Determine optimal installation mode
        profile.installation_mode = self._determine_installation_mode(profile)
        
        self._cache = profile
        return profile
    
    def _detect_system_info(self) -> SystemInfo:
        """Detect basic system information."""
        try:
            # Get system details
            os_name = platform.system()
            os_version = platform.release()
            architecture = platform.machine()
            
            # Python information
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            python_executable = sys.executable
            
            # Virtual environment detection
            in_venv = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )
            venv_path = None
            if in_venv:
                venv_path = os.environ.get('VIRTUAL_ENV', sys.prefix)
            
            # Shell detection
            shell = self._detect_shell()
            
            # Terminal capabilities
            terminal = os.environ.get('TERM', 'unknown')
            supports_colors = self._supports_colors()
            supports_unicode = self._supports_unicode()
            
            # Hardware info
            cpu_count = os.cpu_count() or 1
            memory_gb = self._get_memory_gb()
            
            return SystemInfo(
                os_name=os_name,
                os_version=os_version, 
                architecture=architecture,
                python_version=python_version,
                python_executable=python_executable,
                in_venv=in_venv,
                venv_path=venv_path,
                shell=shell,
                terminal=terminal,
                supports_colors=supports_colors,
                supports_unicode=supports_unicode,
                cpu_count=cpu_count,
                memory_gb=memory_gb
            )
            
        except Exception as e:
            logger.warning(f"Error detecting system info: {e}")
            return SystemInfo(
                os_name="unknown",
                os_version="unknown",
                architecture="unknown", 
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                python_executable=sys.executable,
                in_venv=False
            )
    
    def _detect_development_tools(self) -> DevelopmentTools:
        """Detect available development tools and services."""
        tools = DevelopmentTools()
        
        # Check for common command line tools
        tools.git_available = shutil.which("git") is not None
        tools.docker_available = self._check_docker()
        tools.node_available = shutil.which("node") is not None
        tools.npm_available = shutil.which("npm") is not None
        tools.pip_available = shutil.which("pip") is not None
        tools.conda_available = shutil.which("conda") is not None
        
        # Check Ollama specifically
        tools.ollama_available = shutil.which("ollama") is not None
        if tools.ollama_available:
            tools.ollama_running, tools.ollama_models = self._check_ollama_status()
            
        return tools
    
    def _detect_project_context(self) -> ProjectContext:
        """Detect project context and type."""
        cwd = Path.cwd()
        context = ProjectContext()
        
        # Check for common project files
        context.is_git_repo = (cwd / ".git").exists()
        context.has_requirements_txt = (cwd / "requirements.txt").exists()
        context.has_setup_py = (cwd / "setup.py").exists() 
        context.has_pyproject_toml = (cwd / "pyproject.toml").exists()
        context.has_package_json = (cwd / "package.json").exists()
        context.has_dockerfile = (cwd / "Dockerfile").exists()
        context.has_docker_compose = (cwd / "docker-compose.yml").exists() or (cwd / "docker-compose.yaml").exists()
        
        # Determine project type
        if context.has_package_json:
            context.detected_project_type = "nodejs"
        elif context.has_pyproject_toml or context.has_setup_py:
            context.detected_project_type = "python"
        elif context.has_dockerfile:
            context.detected_project_type = "containerized"
        elif context.is_git_repo:
            context.detected_project_type = "git"
        else:
            context.detected_project_type = "unknown"
            
        return context
    
    def _detect_api_keys(self) -> APIKeysDetection:
        """Detect available API keys from environment variables."""
        detection = APIKeysDetection()
        
        # Map environment variables to providers
        key_mappings = {
            'openai': ['OPENAI_API_KEY', 'OPENAI_KEY'],
            'anthropic': ['ANTHROPIC_API_KEY', 'ANTHROPIC_KEY'], 
            'google_ai': ['GOOGLE_API_KEY', 'GOOGLE_AI_API_KEY', 'GEMINI_API_KEY'],
            'openrouter': ['OPENROUTER_API_KEY', 'OPENROUTER_KEY'],
            'ollama': ['OLLAMA_API_KEY', 'OLLAMA_KEY']
        }
        
        for provider, env_vars in key_mappings.items():
            for env_var in env_vars:
                value = os.getenv(env_var)
                if value and len(value.strip()) > 10:  # Basic validation
                    setattr(detection, f"{provider}_key", value)
                    detection.detected_providers.add(provider)
                    break
                    
        return detection
    
    def _detect_mcp_servers(self) -> MCPServers:
        """Detect existing MCP server configurations."""
        servers = MCPServers()
        
        # Look for common MCP config locations
        possible_configs = [
            Path.home() / ".mcp" / "config.json",
            Path.home() / ".agentsmcp" / "config.yaml",
            Path.cwd() / "mcp.json",
            Path.cwd() / ".mcp.json",
            Path.cwd() / "agentsmcp.yaml"
        ]
        
        for config_path in possible_configs:
            if config_path.exists():
                servers.config_files.append(config_path)
                
        # TODO: Detect running MCP servers (would require MCP registry/discovery)
        # servers.running_servers = self._scan_running_mcp_servers()
        
        return servers
    
    def _determine_installation_mode(self, profile: EnvironmentProfile) -> str:
        """Determine optimal installation mode based on environment."""
        # Production environment indicators
        if any([
            'PRODUCTION' in os.environ,
            'PROD' in os.environ,
            'NODE_ENV' in os.environ and os.environ['NODE_ENV'] == 'production',
            profile.project.has_dockerfile,
            profile.system.in_venv and 'site-packages' in str(profile.system.venv_path or '')
        ]):
            return "production"
            
        # Development environment indicators  
        if any([
            profile.project.is_git_repo,
            profile.project.has_requirements_txt,
            profile.tools.git_available,
            profile.system.in_venv
        ]):
            return "development"
            
        # Containerized environment
        if any([
            profile.project.has_dockerfile,
            profile.project.has_docker_compose,
            profile.tools.docker_available,
            'DOCKER' in os.environ
        ]):
            return "containerized"
            
        return "auto"
    
    def _detect_shell(self) -> str:
        """Detect the current shell."""
        shell_env = os.environ.get('SHELL', '')
        if shell_env:
            return Path(shell_env).name
        
        # Windows detection
        if platform.system() == 'Windows':
            if 'POWERSHELL' in os.environ.get('PSModulePath', ''):
                return 'powershell'
            return 'cmd'
            
        return 'unknown'
    
    def _supports_colors(self) -> bool:
        """Check if terminal supports colors."""
        if platform.system() == 'Windows':
            # Modern Windows Terminal and PowerShell support colors
            return 'WT_SESSION' in os.environ or 'TERM_PROGRAM' in os.environ
        
        term = os.environ.get('TERM', '')
        return any(color_term in term.lower() for color_term in [
            'color', 'xterm', 'screen', 'tmux', 'ansi'
        ])
    
    def _supports_unicode(self) -> bool:
        """Check if terminal supports Unicode."""
        try:
            print('✓', end='', file=open(os.devnull, 'w'))
            return True
        except UnicodeError:
            return False
    
    def _get_memory_gb(self) -> float:
        """Get system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback methods for systems without psutil
            if platform.system() == 'Linux':
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemTotal:'):
                                kb = int(line.split()[1])
                                return kb / (1024**2)
                except:
                    pass
            return 0.0
    
    def _check_docker(self) -> bool:
        """Check if Docker is available and working."""
        if not shutil.which("docker"):
            return False
            
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False
    
    def _check_ollama_status(self) -> tuple[bool, List[str]]:
        """Check if Ollama is running and get available models."""
        try:
            # Check if Ollama server is running
            response = urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
            data = json.loads(response.read().decode())
            models = [m["name"] for m in data.get("models", []) if "name" in m]
            return True, models
        except (URLError, HTTPError, json.JSONDecodeError, Exception):
            return False, []


# For backwards compatibility and convenience
def detect_environment(verbose: bool = False) -> EnvironmentProfile:
    """Convenience function for quick environment detection."""
    detector = EnvironmentDetector(verbose=verbose)
    return detector.detect_environment()


# Import time at module level 
import time