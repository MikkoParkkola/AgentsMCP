"""
Quality Gate System - Advanced Code Quality and Safety Controls

This module implements comprehensive quality gates and safety measures to prevent
code breakage when AgentsMCP modifies its own codebase or other critical systems.
"""

import ast
import json
import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import shutil
import tempfile

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality levels for different types of changes."""
    CRITICAL = "critical"  # Core system files, configs
    HIGH = "high"         # Important functionality
    MEDIUM = "medium"     # Standard features
    LOW = "low"           # Documentation, comments


class QualityGateResult(Enum):
    """Results of quality gate evaluation."""
    PASS = "pass"
    FAIL = "fail" 
    WARNING = "warning"


@dataclass
class QualityCheck:
    """Represents a single quality check."""
    name: str
    description: str
    level: QualityLevel
    result: QualityGateResult
    details: str
    suggestions: List[str]


@dataclass
class QualityGateReport:
    """Report from running quality gates."""
    overall_result: QualityGateResult
    checks: List[QualityCheck]
    critical_issues: int
    warnings: int
    file_path: Optional[str] = None


class CodeSafetyAnalyzer:
    """Analyzes code for potential safety issues."""
    
    def __init__(self):
        self.critical_patterns = [
            # Dangerous operations
            "shutil.rmtree", "os.remove", "os.rmdir", 
            "subprocess.call", "subprocess.run", "os.system",
            "exec(", "eval(", "__import__",
            # File operations on critical files
            "runtime_config.py", "settings.py", "__init__.py",
            # Database operations
            "DROP TABLE", "DELETE FROM", "TRUNCATE"
        ]
        
        self.critical_paths = {
            "src/agentsmcp/runtime_config.py",
            "src/agentsmcp/settings.py", 
            "src/agentsmcp/cli.py",
            "src/agentsmcp/__init__.py"
        }
    
    def analyze_code_change(self, file_path: str, content: str) -> List[QualityCheck]:
        """Analyze code changes for safety issues."""
        checks = []
        
        # Check if modifying critical files
        if any(critical_path in file_path for critical_path in self.critical_paths):
            checks.append(QualityCheck(
                name="Critical File Modification",
                description=f"Modifying critical system file: {file_path}",
                level=QualityLevel.CRITICAL,
                result=QualityGateResult.WARNING,
                details="This file is critical to system operation",
                suggestions=[
                    "Create backup before modification",
                    "Test changes in isolated environment",
                    "Have rollback plan ready"
                ]
            ))
        
        # Check for dangerous patterns
        dangerous_found = []
        for pattern in self.critical_patterns:
            if pattern in content:
                dangerous_found.append(pattern)
        
        if dangerous_found:
            checks.append(QualityCheck(
                name="Dangerous Operations Detected",
                description="Found potentially dangerous operations in code",
                level=QualityLevel.CRITICAL,
                result=QualityGateResult.FAIL,
                details=f"Dangerous patterns: {', '.join(dangerous_found)}",
                suggestions=[
                    "Review dangerous operations for necessity",
                    "Add safety checks and validation",
                    "Consider safer alternatives",
                    "Add comprehensive error handling"
                ]
            ))
        
        # Syntax validation for Python files
        if file_path.endswith('.py'):
            syntax_check = self.validate_python_syntax(content, file_path)
            if syntax_check:
                checks.append(syntax_check)
                
        return checks
    
    def validate_python_syntax(self, content: str, file_path: str) -> Optional[QualityCheck]:
        """Validate Python syntax."""
        try:
            ast.parse(content)
            return None
        except SyntaxError as e:
            return QualityCheck(
                name="Syntax Error",
                description="Python syntax error detected",
                level=QualityLevel.CRITICAL,
                result=QualityGateResult.FAIL,
                details=f"Syntax error at line {e.lineno}: {e.msg}",
                suggestions=[
                    "Fix syntax error before proceeding",
                    "Use linter to catch syntax issues",
                    "Review code formatting"
                ]
            )


class BackupManager:
    """Manages backups of critical files."""
    
    def __init__(self, backup_dir: Optional[Path] = None):
        if backup_dir is None:
            backup_dir = Path.home() / ".agentsmcp" / "backups"
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, file_path: Path) -> Path:
        """Create backup of a file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot backup non-existent file: {file_path}")
            
        backup_name = f"{file_path.name}.backup.{int(Path().stat().st_mtime)}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    
    def restore_backup(self, original_path: Path, backup_path: Path):
        """Restore file from backup."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
        shutil.copy2(backup_path, original_path)
        logger.info(f"Restored {original_path} from backup {backup_path}")


class QualityGateSystem:
    """Main quality gate system coordinator."""
    
    def __init__(self):
        self.safety_analyzer = CodeSafetyAnalyzer()
        self.backup_manager = BackupManager()
        self.git_integration = GitIntegration()
        
    def evaluate_change(self, file_path: str, content: str) -> QualityGateReport:
        """Evaluate a proposed code change through all quality gates."""
        checks = []
        
        # Run safety analysis
        safety_checks = self.safety_analyzer.analyze_code_change(file_path, content)
        checks.extend(safety_checks)
        
        # Run additional checks based on file type
        if file_path.endswith('.py'):
            checks.extend(self._run_python_checks(file_path, content))
        elif file_path.endswith('.json'):
            checks.extend(self._run_json_checks(file_path, content))
            
        # Determine overall result
        critical_issues = sum(1 for check in checks if check.result == QualityGateResult.FAIL)
        warnings = sum(1 for check in checks if check.result == QualityGateResult.WARNING)
        
        if critical_issues > 0:
            overall_result = QualityGateResult.FAIL
        elif warnings > 0:
            overall_result = QualityGateResult.WARNING
        else:
            overall_result = QualityGateResult.PASS
            
        return QualityGateReport(
            overall_result=overall_result,
            checks=checks,
            critical_issues=critical_issues,
            warnings=warnings,
            file_path=file_path
        )
    
    def _run_python_checks(self, file_path: str, content: str) -> List[QualityCheck]:
        """Run Python-specific quality checks."""
        checks = []
        
        # Check imports
        try:
            tree = ast.parse(content)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        
            # Check for risky imports
            risky_imports = {'os', 'subprocess', 'shutil', 'sys'}
            found_risky = set(imports) & risky_imports
            
            if found_risky:
                checks.append(QualityCheck(
                    name="Risky Imports",
                    description="Found imports that could be dangerous",
                    level=QualityLevel.HIGH,
                    result=QualityGateResult.WARNING,
                    details=f"Risky imports: {', '.join(found_risky)}",
                    suggestions=[
                        "Review usage of risky imports",
                        "Add proper error handling",
                        "Consider safer alternatives"
                    ]
                ))
        except SyntaxError:
            pass  # Syntax error already caught by safety analyzer
            
        return checks
    
    def _run_json_checks(self, file_path: str, content: str) -> List[QualityCheck]:
        """Run JSON-specific quality checks."""
        checks = []
        
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            checks.append(QualityCheck(
                name="JSON Syntax Error",
                description="Invalid JSON format",
                level=QualityLevel.HIGH,
                result=QualityGateResult.FAIL,
                details=f"JSON error: {e.msg}",
                suggestions=[
                    "Fix JSON syntax error",
                    "Validate JSON structure",
                    "Use JSON linter"
                ]
            ))
            
        return checks
    
    def safe_modify_file(self, file_path: Path, new_content: str) -> bool:
        """Safely modify a file with quality gates and backup."""
        try:
            # Step 1: Evaluate the change
            report = self.evaluate_change(str(file_path), new_content)
            
            if report.overall_result == QualityGateResult.FAIL:
                logger.error(f"Quality gate FAILED for {file_path}")
                for check in report.checks:
                    if check.result == QualityGateResult.FAIL:
                        logger.error(f"  {check.name}: {check.details}")
                return False
            
            # Step 2: Create backup if file exists
            backup_path = None
            if file_path.exists():
                backup_path = self.backup_manager.create_backup(file_path)
                
            # Step 3: Create Git stash if in Git repo
            git_stash_created = self.git_integration.create_safety_stash(file_path.parent)
            
            # Step 4: Write the new content
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    
                logger.info(f"Successfully modified {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to write file {file_path}: {e}")
                
                # Restore from backup if possible
                if backup_path:
                    try:
                        self.backup_manager.restore_backup(file_path, backup_path)
                        logger.info(f"Restored {file_path} from backup")
                    except Exception as restore_error:
                        logger.error(f"Failed to restore backup: {restore_error}")
                        
                return False
                
        except Exception as e:
            logger.error(f"Quality gate system error for {file_path}: {e}")
            return False


class GitIntegration:
    """Git integration for safety operations."""
    
    def create_safety_stash(self, repo_path: Path) -> bool:
        """Create a git stash for safety before dangerous operations."""
        try:
            result = subprocess.run(
                ['git', 'stash', 'push', '-m', 'AgentsMCP safety stash'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Could not create git stash: {e}")
            return False
    
    def is_git_repo(self, path: Path) -> bool:
        """Check if path is in a git repository."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=path,
                capture_output=True
            )
            return result.returncode == 0
        except Exception:
            return False


# Global quality gate system
_quality_gate_system = None


def get_quality_gate_system() -> QualityGateSystem:
    """Get the global quality gate system instance."""
    global _quality_gate_system
    if _quality_gate_system is None:
        _quality_gate_system = QualityGateSystem()
    return _quality_gate_system