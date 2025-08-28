"""
Template management subsystem for AgentsMCP pipelines.

Responsibilities
----------------
* Discover built-in and user-provided pipeline templates.
* Load a template file, render Jinja2 placeholders with a user supplied
  context, and return a Python dict.
* Validate the rendered pipeline against the schema defined in
  ``agentsmcp.pipeline.schema``.
* Provide helper utilities for listing available templates.

The public API is a single class: :class:`TemplateManager`.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Dict, List, Mapping, Optional, Any

import yaml
from jinja2 import Environment, StrictUndefined, TemplateError
from pydantic import ValidationError

from agentsmcp.pipeline.schema import PipelineSpec


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class TemplateErrorBase(Exception):
    """Base class for all template-related errors."""


class TemplateNotFoundError(TemplateErrorBase):
    """Raised when a requested template cannot be located."""


class TemplateRenderingError(TemplateErrorBase):
    """Raised when Jinja2 fails to render a template."""


class TemplateValidationError(TemplateErrorBase):
    """Raised when the rendered pipeline does not conform to the schema."""

    def __init__(self, errors: List[dict]) -> None:
        message = "Pipeline validation failed:\n"
        for err in errors:
            loc = " -> ".join(str(loc) for loc in err.get("loc", []))
            message += f" - {err.get('msg', 'Unknown error')} (path: {loc})\n"
        super().__init__(message)
        self.errors = errors


# --------------------------------------------------------------------------- #
# Core manager
# --------------------------------------------------------------------------- #
class TemplateManager:
    """
    Handles discovery, rendering and validation of pipeline templates.

    Parameters
    ----------
    user_dir : str | pathlib.Path, optional
        Directory where user-overridden templates live. By default the
        project root ``user_templates/pipelines`` is used if it exists.
    builtin_dir : str | pathlib.Path, optional
        Directory that ships with the library (``agentsmcp/templates/pipelines``).
    """

    # Built-in template names for quick reference
    _BUILTIN_TEMPLATES = ["basic", "python-package", "node-app", "data-science"]

    # ------------------------------------------------------------------- #
    # Construction
    # ------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        user_dir: Optional[os.PathLike] = None,
        builtin_dir: Optional[os.PathLike] = None,
    ) -> None:
        self._builtin_dir = pathlib.Path(
            builtin_dir or pathlib.Path(__file__).parent / "pipelines"
        ).resolve()
        
        self._user_dir = pathlib.Path(
            user_dir or pathlib.Path.cwd() / "user_templates" / "pipelines"
        ).resolve()

        # Cache of available template names → absolute path
        self._cache: Dict[str, pathlib.Path] = {}
        self._discover_templates()

    # ------------------------------------------------------------------- #
    # Discovery helpers
    # ------------------------------------------------------------------- #
    def _discover_templates(self) -> None:
        """
        Populate ``self._cache`` with all template names found in the user
        directory (if it exists) *first*, then the built-in directory.
        The key is the file stem (e.g. ``python-package``) and the value is
        the absolute ``Path`` to the YAML file.
        """
        self._cache.clear()
        
        # 1. user templates (override)
        if self._user_dir.is_dir():
            for p in self._user_dir.glob("*.yaml"):
                self._cache[p.stem] = p
            for p in self._user_dir.glob("*.yml"):
                if p.stem not in self._cache:  # .yaml takes precedence
                    self._cache[p.stem] = p

        # 2. built-in templates (fallback if not overridden)
        if self._builtin_dir.is_dir():
            for p in self._builtin_dir.glob("*.yaml"):
                if p.stem not in self._cache:
                    self._cache[p.stem] = p
            for p in self._builtin_dir.glob("*.yml"):
                if p.stem not in self._cache:
                    self._cache[p.stem] = p

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #
    def list_templates(self) -> List[str]:
        """
        Return a sorted list of available template identifiers.
        """
        return sorted(self._cache.keys())

    def get_template_info(self, name: str) -> Dict[str, Any]:
        """
        Get template metadata without full loading.
        
        Returns basic info like name, description from the template.
        """
        if name not in self._cache:
            return {
                "name": name,
                "description": f"Template '{name}' not found",
                "available": False
            }
        
        try:
            template_path = self._cache[name]
            raw_text = template_path.read_text(encoding="utf-8")
            
            # Parse just the header to get basic info
            template_data = yaml.safe_load(raw_text) or {}
            
            return {
                "name": name,
                "description": template_data.get("description", "No description available"),
                "stages": len(template_data.get("stages", [])),
                "available": True,
                "path": str(template_path)
            }
        except Exception:
            return {
                "name": name,
                "description": "Error reading template",
                "available": False
            }

    def get_template_path(self, name: str) -> pathlib.Path:
        """
        Return the absolute path of the template ``name``.
        Raises :class:`TemplateNotFoundError` if the name is unknown.
        """
        try:
            return self._cache[name]
        except KeyError as exc:
            available = ", ".join(self.list_templates())
            raise TemplateNotFoundError(
                f"Template '{name}' not found. Available templates: {available}"
            ) from exc

    def load_template(
        self,
        name: str,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Load, render and validate a pipeline template.

        Parameters
        ----------
        name :
            Identifier of the desired template (file stem without extension).
        variables :
            Mapping of placeholder names → values fed to Jinja2.
            Missing variables cause a ``TemplateRenderingError`` because we
            use ``StrictUndefined``.

        Returns
        -------
        dict
            The fully-rendered pipeline configuration ready to be written
            as YAML/JSON.

        Raises
        ------
        TemplateNotFoundError
            If the template does not exist.
        TemplateRenderingError
            If Jinja2 encounters an undefined variable or syntax error.
        TemplateValidationError
            If the final dict does not satisfy the schema.
        """
        tmpl_path = self.get_template_path(name)

        # -----------------------------------------------------------------
        # 1️⃣ Read raw YAML (as a Jinja2 template)
        # -----------------------------------------------------------------
        try:
            raw_text = tmpl_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TemplateErrorBase(f"Failed to read template file {tmpl_path}") from exc

        # -----------------------------------------------------------------
        # 2️⃣ Render Jinja2
        # -----------------------------------------------------------------
        env = Environment(undefined=StrictUndefined, autoescape=False)
        try:
            rendered_text = env.from_string(raw_text).render(**(variables or {}))
        except TemplateError as exc:
            raise TemplateRenderingError(f"Error rendering template '{name}': {exc}") from exc

        # -----------------------------------------------------------------
        # 3️⃣ Parse YAML => Python dict
        # -----------------------------------------------------------------
        try:
            pipeline_dict = yaml.safe_load(rendered_text) or {}
        except yaml.YAMLError as exc:
            raise TemplateRenderingError(
                f"YAML syntax error after rendering template '{name}': {exc}"
            ) from exc

        # -----------------------------------------------------------------
        # 4️⃣ Validate against Pydantic schema
        # -----------------------------------------------------------------
        try:
            # Use Pydantic model for validation
            pipeline_spec = PipelineSpec(**pipeline_dict)
            # Convert back to dict for compatibility
            return pipeline_spec.model_dump()
        except ValidationError as exc:
            errors = []
            for error in exc.errors():
                errors.append({
                    "loc": error["loc"],
                    "msg": error["msg"],
                    "type": error["type"]
                })
            raise TemplateValidationError(errors) from exc

    def create_pipeline_from_template(
        self,
        template_name: str,
        output_path: os.PathLike,
        *,
        variables: Mapping[str, Any] | None = None,
        indent: int = 2,
    ) -> Dict[str, Any]:
        """
        Complete pipeline creation workflow: load template, render, validate, and save.
        
        Parameters
        ----------
        template_name : str
            Name of the template to use
        output_path : os.PathLike
            Where to save the generated pipeline YAML
        variables : dict, optional
            Template variables for Jinja2 rendering
        indent : int
            YAML indentation level
            
        Returns
        -------
        dict
            The generated pipeline configuration
        """
        # Load and render the template
        pipeline_dict = self.load_template(template_name, variables=variables)
        
        # Save to file
        self.write_pipeline(pipeline_dict, output_path, indent=indent)
        
        return pipeline_dict

    def write_pipeline(
        self,
        pipeline_dict: Dict[str, Any],
        output_path: os.PathLike,
        *,
        indent: int = 2,
    ) -> None:
        """
        Serialise ``pipeline_dict`` as a YAML file to ``output_path``.
        The function creates parent directories if required.

        Parameters
        ----------
        pipeline_dict :
            The pipeline configuration to write.
        output_path :
            Destination file (Path, str, etc.)
        indent :
            Number of spaces for YAML indentation (default 2).
        """
        out_path = pathlib.Path(output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with out_path.open("w", encoding="utf-8") as fp:
                yaml.dump(
                    pipeline_dict, 
                    fp, 
                    indent=indent, 
                    default_flow_style=False, 
                    sort_keys=False,
                    allow_unicode=True
                )
        except OSError as exc:
            raise TemplateErrorBase(f"Failed to write pipeline to {out_path}") from exc

    def validate_template_file(self, template_path: os.PathLike) -> Dict[str, Any]:
        """
        Validate a template file directly (without rendering).
        
        Useful for checking template syntax before using it.
        """
        tmpl_path = pathlib.Path(template_path)
        
        if not tmpl_path.exists():
            raise TemplateNotFoundError(f"Template file not found: {tmpl_path}")
        
        try:
            raw_text = tmpl_path.read_text(encoding="utf-8")
            # Parse YAML to check basic syntax
            yaml.safe_load(raw_text)
            
            return {
                "valid": True,
                "path": str(tmpl_path),
                "message": "Template syntax is valid"
            }
        except yaml.YAMLError as exc:
            return {
                "valid": False,
                "path": str(tmpl_path),
                "message": f"YAML syntax error: {exc}"
            }
        except Exception as exc:
            return {
                "valid": False,
                "path": str(tmpl_path), 
                "message": f"Error reading template: {exc}"
            }


# --------------------------------------------------------------------------- #
# Convenience functions
# --------------------------------------------------------------------------- #
def get_default_template_manager() -> TemplateManager:
    """Get a TemplateManager instance with default settings."""
    return TemplateManager()


def list_available_templates() -> List[str]:
    """Quick function to list all available templates."""
    manager = get_default_template_manager()
    return manager.list_templates()


def create_pipeline_from_template(
    template_name: str,
    output_path: os.PathLike,
    variables: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Convenience function to create a pipeline from a template.
    
    This is the most common use case - create a pipeline YAML file
    from a template with variables.
    """
    manager = get_default_template_manager()
    return manager.create_pipeline_from_template(
        template_name, output_path, variables=variables
    )