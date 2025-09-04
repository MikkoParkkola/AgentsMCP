"""Project detection utilities for enhanced preprocessing context."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # fallback for older Python versions
    except ImportError:
        tomllib = None

logger = logging.getLogger(__name__)


class ProjectDetector:
    """Detects project type and extracts relevant context information."""
    
    @staticmethod
    def detect_project_context(directory: str) -> Dict[str, Any]:
        """Detect and summarize current project context from directory."""
        try:
            project_path = Path(directory).resolve()
            if not project_path.exists() or not project_path.is_dir():
                return {"error": f"Directory does not exist: {directory}"}
            
            context = {
                "directory": str(project_path),
                "project_type": "unknown",
                "project_name": project_path.name,
                "description": "",
                "key_files": [],
                "languages": [],
                "frameworks": [],
                "purpose": "",
                "structure_summary": ""
            }
            
            # Check for common project files
            project_files = ProjectDetector._scan_project_files(project_path)
            context["key_files"] = project_files
            
            # Detect project type and extract metadata
            context.update(ProjectDetector._detect_python_project(project_path))
            if context["project_type"] == "unknown":
                context.update(ProjectDetector._detect_nodejs_project(project_path))
            if context["project_type"] == "unknown":
                context.update(ProjectDetector._detect_rust_project(project_path))
            if context["project_type"] == "unknown":
                context.update(ProjectDetector._detect_general_project(project_path))
            
            # Get README summary if available
            readme_summary = ProjectDetector._extract_readme_summary(project_path)
            if readme_summary:
                context["description"] = readme_summary
            
            # Generate structure summary
            context["structure_summary"] = ProjectDetector._generate_structure_summary(project_path)
            
            return context
            
        except Exception as e:
            logger.error(f"Error detecting project context: {e}")
            return {
                "directory": directory,
                "error": f"Failed to detect project context: {str(e)}"
            }
    
    @staticmethod
    def _scan_project_files(project_path: Path) -> List[str]:
        """Scan for important project files."""
        important_files = [
            "README.md", "README.rst", "README.txt",
            "pyproject.toml", "setup.py", "requirements.txt", "Pipfile",
            "package.json", "package-lock.json", "yarn.lock",
            "Cargo.toml", "Cargo.lock",
            ".gitignore", ".git", "LICENSE", "CHANGELOG.md",
            "Dockerfile", "docker-compose.yml",
            "Makefile", "CMakeLists.txt"
        ]
        
        found_files = []
        for file_name in important_files:
            if (project_path / file_name).exists():
                found_files.append(file_name)
        
        # Check for common directories
        common_dirs = ["src", "lib", "app", "tests", "docs", ".github"]
        for dir_name in common_dirs:
            if (project_path / dir_name).is_dir():
                found_files.append(f"{dir_name}/")
        
        return found_files
    
    @staticmethod
    def _detect_python_project(project_path: Path) -> Dict[str, Any]:
        """Detect Python project and extract metadata."""
        context = {}
        
        # Check for pyproject.toml
        pyproject_path = project_path / "pyproject.toml"
        if pyproject_path.exists() and tomllib is not None:
            try:
                with open(pyproject_path, 'rb') as f:
                    pyproject = tomllib.load(f)
                
                context["project_type"] = "python"
                context["languages"] = ["python"]
                
                # Extract project info
                if "project" in pyproject:
                    proj = pyproject["project"]
                    context["project_name"] = proj.get("name", context.get("project_name", ""))
                    context["description"] = proj.get("description", "")
                    
                # Extract dependencies for framework detection
                deps = pyproject.get("project", {}).get("dependencies", [])
                context["frameworks"] = ProjectDetector._detect_python_frameworks(deps)
                
                # Check build system
                if "build-system" in pyproject:
                    build_requires = pyproject["build-system"].get("requires", [])
                    if any("poetry" in req for req in build_requires):
                        context["frameworks"].append("poetry")
                    elif any("setuptools" in req for req in build_requires):
                        context["frameworks"].append("setuptools")
                
                return context
            except Exception as e:
                logger.warning(f"Error reading pyproject.toml: {e}")
        
        # Check for setup.py
        setup_path = project_path / "setup.py"
        if setup_path.exists():
            context["project_type"] = "python"
            context["languages"] = ["python"]
            context["frameworks"] = ["setuptools"]
            return context
        
        # Check for requirements.txt
        req_path = project_path / "requirements.txt"
        if req_path.exists():
            try:
                with open(req_path, 'r', encoding='utf-8') as f:
                    deps = f.read().splitlines()
                context["project_type"] = "python"
                context["languages"] = ["python"]
                context["frameworks"] = ProjectDetector._detect_python_frameworks(deps)
                return context
            except Exception as e:
                logger.warning(f"Error reading requirements.txt: {e}")
        
        return context
    
    @staticmethod
    def _detect_python_frameworks(dependencies: List[str]) -> List[str]:
        """Detect Python frameworks from dependency list."""
        frameworks = []
        dep_str = " ".join(str(dep).lower() for dep in dependencies)
        
        framework_patterns = {
            "fastapi": ["fastapi"],
            "flask": ["flask"],
            "django": ["django"],
            "pytest": ["pytest"],
            "streamlit": ["streamlit"],
            "asyncio": ["asyncio", "aiohttp"],
            "pydantic": ["pydantic"],
            "sqlalchemy": ["sqlalchemy"],
            "numpy": ["numpy"],
            "pandas": ["pandas"],
            "pytorch": ["torch", "pytorch"],
            "tensorflow": ["tensorflow"]
        }
        
        for framework, patterns in framework_patterns.items():
            if any(pattern in dep_str for pattern in patterns):
                frameworks.append(framework)
        
        return frameworks
    
    @staticmethod
    def _detect_nodejs_project(project_path: Path) -> Dict[str, Any]:
        """Detect Node.js project and extract metadata."""
        context = {}
        
        package_path = project_path / "package.json"
        if package_path.exists():
            try:
                with open(package_path, 'r', encoding='utf-8') as f:
                    package = json.load(f)
                
                context["project_type"] = "nodejs"
                context["languages"] = ["javascript"]
                
                context["project_name"] = package.get("name", context.get("project_name", ""))
                context["description"] = package.get("description", "")
                
                # Detect TypeScript
                deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
                if "typescript" in deps or "@types/" in str(deps):
                    context["languages"].append("typescript")
                
                # Detect frameworks
                context["frameworks"] = ProjectDetector._detect_js_frameworks(deps)
                
                return context
            except Exception as e:
                logger.warning(f"Error reading package.json: {e}")
        
        return context
    
    @staticmethod
    def _detect_js_frameworks(dependencies: Dict[str, str]) -> List[str]:
        """Detect JavaScript/TypeScript frameworks from dependencies."""
        frameworks = []
        dep_names = list(dependencies.keys())
        dep_str = " ".join(dep_names).lower()
        
        framework_patterns = {
            "react": ["react"],
            "vue": ["vue"],
            "angular": ["@angular/"],
            "express": ["express"],
            "nest": ["@nestjs/"],
            "next": ["next"],
            "nuxt": ["nuxt"],
            "svelte": ["svelte"],
            "electron": ["electron"],
            "jest": ["jest"],
            "webpack": ["webpack"],
            "vite": ["vite"]
        }
        
        for framework, patterns in framework_patterns.items():
            if any(pattern in dep_str for pattern in patterns):
                frameworks.append(framework)
        
        return frameworks
    
    @staticmethod
    def _detect_rust_project(project_path: Path) -> Dict[str, Any]:
        """Detect Rust project and extract metadata."""
        context = {}
        
        cargo_path = project_path / "Cargo.toml"
        if cargo_path.exists() and tomllib is not None:
            try:
                with open(cargo_path, 'rb') as f:
                    cargo = tomllib.load(f)
                
                context["project_type"] = "rust"
                context["languages"] = ["rust"]
                
                if "package" in cargo:
                    pkg = cargo["package"]
                    context["project_name"] = pkg.get("name", context.get("project_name", ""))
                    context["description"] = pkg.get("description", "")
                
                # Detect frameworks from dependencies
                deps = cargo.get("dependencies", {})
                context["frameworks"] = ProjectDetector._detect_rust_frameworks(deps)
                
                return context
            except Exception as e:
                logger.warning(f"Error reading Cargo.toml: {e}")
        
        return context
    
    @staticmethod
    def _detect_rust_frameworks(dependencies: Dict[str, Any]) -> List[str]:
        """Detect Rust frameworks from dependencies."""
        frameworks = []
        dep_names = list(dependencies.keys())
        
        framework_patterns = {
            "tokio": ["tokio"],
            "actix": ["actix-web", "actix"],
            "axum": ["axum"],
            "warp": ["warp"],
            "serde": ["serde"],
            "clap": ["clap"],
            "diesel": ["diesel"]
        }
        
        for framework, patterns in framework_patterns.items():
            if any(pattern in dep_names for pattern in patterns):
                frameworks.append(framework)
        
        return frameworks
    
    @staticmethod
    def _detect_general_project(project_path: Path) -> Dict[str, Any]:
        """Detect general project information when specific type unknown."""
        context = {}
        
        # Check for git repository
        if (project_path / ".git").exists():
            context["project_type"] = "git"
        
        # Check for Docker
        if (project_path / "Dockerfile").exists() or (project_path / "docker-compose.yml").exists():
            context["frameworks"] = context.get("frameworks", []) + ["docker"]
        
        # Check for Makefile
        if (project_path / "Makefile").exists():
            context["frameworks"] = context.get("frameworks", []) + ["make"]
        
        return context
    
    @staticmethod
    def _extract_readme_summary(project_path: Path) -> str:
        """Extract summary from README file."""
        readme_files = ["README.md", "README.rst", "README.txt", "readme.md"]
        
        for readme_file in readme_files:
            readme_path = project_path / readme_file
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract first meaningful paragraph
                    lines = content.split('\n')
                    summary_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('#'):
                            continue
                        if len(line) > 20 and not line.startswith('[!['):
                            summary_lines.append(line)
                            if len(' '.join(summary_lines)) > 200:
                                break
                    
                    summary = ' '.join(summary_lines)[:300]
                    return summary if len(summary) > 10 else ""
                    
                except Exception as e:
                    logger.warning(f"Error reading README: {e}")
        
        return ""
    
    @staticmethod
    def _generate_structure_summary(project_path: Path) -> str:
        """Generate a brief project structure summary."""
        try:
            structure_parts = []
            
            # Check main source directories
            src_dirs = ["src", "lib", "app", "components"]
            found_src = [d for d in src_dirs if (project_path / d).is_dir()]
            if found_src:
                structure_parts.append(f"Source: {', '.join(found_src)}")
            
            # Check test directories
            test_dirs = ["tests", "test", "__tests__"]
            found_tests = [d for d in test_dirs if (project_path / d).is_dir()]
            if found_tests:
                structure_parts.append(f"Tests: {', '.join(found_tests)}")
            
            # Check documentation
            doc_dirs = ["docs", "documentation"]
            found_docs = [d for d in doc_dirs if (project_path / d).is_dir()]
            if found_docs:
                structure_parts.append(f"Docs: {', '.join(found_docs)}")
            
            return " | ".join(structure_parts) if structure_parts else "Standard project layout"
            
        except Exception:
            return "Unknown structure"


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation)."""
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def format_project_context(project_context: Dict[str, Any]) -> str:
    """Format project context for preprocessing prompt."""
    if "error" in project_context:
        return f"PROJECT CONTEXT ERROR: {project_context['error']}"
    
    lines = [
        f"CURRENT DIRECTORY CONTEXT:",
        f"- Working Directory: {project_context['directory']}",
        f"- Project Type: {project_context['project_type'].title()} ({project_context['project_name']})"
    ]
    
    if project_context.get('description'):
        lines.append(f"- Description: {project_context['description']}")
    
    if project_context.get('languages'):
        lines.append(f"- Languages: {', '.join(project_context['languages'])}")
    
    if project_context.get('frameworks'):
        lines.append(f"- Frameworks: {', '.join(project_context['frameworks'])}")
    
    if project_context.get('key_files'):
        key_files = project_context['key_files'][:10]  # Limit to first 10
        lines.append(f"- Key Files: {', '.join(key_files)}")
    
    if project_context.get('structure_summary'):
        lines.append(f"- Structure: {project_context['structure_summary']}")
    
    return '\n'.join(lines)


def format_structured_project_context(project_context: Dict[str, Any]) -> Dict[str, Any]:
    """Format project context for structured prompt system."""
    if "error" in project_context:
        return {"error": project_context['error']}
    
    # Build comprehensive project information
    project_info_parts = []
    
    # Basic project information
    project_info_parts.append(f"**PROJECT:** {project_context['project_name']} ({project_context['project_type'].title()})")
    project_info_parts.append(f"**DIRECTORY:** {project_context['directory']}")
    
    if project_context.get('description'):
        project_info_parts.append(f"**DESCRIPTION:** {project_context['description']}")
    
    # Technical details
    tech_details = []
    if project_context.get('languages'):
        tech_details.append(f"Languages: {', '.join(project_context['languages'])}")
    if project_context.get('frameworks'):
        tech_details.append(f"Frameworks: {', '.join(project_context['frameworks'])}")
    
    if tech_details:
        project_info_parts.append(f"**TECH STACK:** {' | '.join(tech_details)}")
    
    # Project structure
    if project_context.get('structure_summary'):
        project_info_parts.append(f"**STRUCTURE:** {project_context['structure_summary']}")
    
    # Key files (limited for readability)
    if project_context.get('key_files'):
        key_files = project_context['key_files'][:8]  # Limit for structured context
        project_info_parts.append(f"**KEY FILES:** {', '.join(key_files)}")
    
    # Return structured format
    return {
        "project_info": "\n".join(project_info_parts),
        "project_type": project_context['project_type'],
        "project_name": project_context['project_name'],
        "directory": project_context['directory'],
        "languages": project_context.get('languages', []),
        "frameworks": project_context.get('frameworks', [])
    }