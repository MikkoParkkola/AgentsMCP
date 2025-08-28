"""
Environment detector module
~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects available API keys, language models, and system characteristics.
Provides helper functions that return dictionaries suitable for configuration.
"""

from __future__ import annotations

import os
import platform
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None


# --------------------------------------------------------------------------- #
#   API KEY detection
# --------------------------------------------------------------------------- #

API_KEY_ENV_VARS = {
    "openai": ["OPENAI_API_KEY", "OPENAI_API_KEY_FILE"],
    "azure": ["AZURE_OPENAI_KEY", "AZURE_OPENAI_KEY_FILE"],
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE"],
}


def _load_from_file(path: Path) -> str:
    try:
        return path.read_text().strip()
    except Exception as exc:
        raise RuntimeError(f"Could not read API key from {path}") from exc


_CACHE: Dict[str, Tuple[float, Any]] = {}
_TTL = 300.0  # seconds


def _cached(key: str) -> Any | None:
    rec = _CACHE.get(key)
    if not rec:
        return None
    ts, val = rec
    if (time.time() - ts) > _TTL:
        _CACHE.pop(key, None)
        return None
    return val


def _store(key: str, val: Any) -> Any:
    _CACHE[key] = (time.time(), val)
    return val


def refresh_env_detector_cache() -> None:
    """Clear detector caches (used by /refresh)."""
    _CACHE.clear()


def detect_api_keys() -> Dict[str, str]:
    """
    Detect presence of API keys in the environment.

    Returns
    -------
    Dict[str, str]
        Mapping provider name → API key string.
    """
    cached = _cached("api_keys")
    if cached is not None:
        return cached
    keys: Dict[str, str] = {}
    for provider, vars_ in API_KEY_ENV_VARS.items():
        for env in vars_:
            val = os.getenv(env)
            if val:
                if env.endswith("_FILE"):
                    try:
                        val = _load_from_file(Path(val))
                    except Exception:
                        continue
                keys[provider] = val
                break
    return _store("api_keys", keys)


# --------------------------------------------------------------------------- #
#   Model detection
# --------------------------------------------------------------------------- #

def list_available_models() -> List[str]:
    """
    Enumerate compatible language models based on available API keys.

    Returns
    -------
    List[str]
        Model identifiers (e.g. ``gpt‑4o-mini``).
    """
    # For demonstration purposes we provide a static mapping.
    # Real projects will interrogate the provider's REST API.
    cached = _cached("models")
    if cached is not None:
        return cached
    providers = detect_api_keys()
    models: List[str] = []

    if "openai" in providers:
        # Example list – in a real system query the OpenAI API
        models.extend(
            [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4o-2024-08-06",
                "gpt-3.5-turbo",
            ]
        )
    if "azure" in providers:
        models.extend(["qwen-large", "claude-3-sonnet-20240229"])
    if "anthropic" in providers:
        models.extend(["claude-3-sonnet-20240229", "claude-3-haiku-20240307"])

    return _store("models", list(dict.fromkeys(models)))  # dedupe while preserving order


# --------------------------------------------------------------------------- #
#   System capability detection
# --------------------------------------------------------------------------- #

def detect_system_info() -> Dict[str, Any]:
    """
    Gather basic system hardware and OS information.

    Returns
    -------
    Dict[str, Any]
        Contains keys such as ``cpu_count``, ``cpu_physical``, ``os``,
        ``memory_mb`` and ``python_version``.
    """
    cached = _cached("sysinfo")
    if cached is not None:
        return cached
    info: Dict[str, Any] = {}
    info["python_version"] = platform.python_version()
    info["os"] = f"{platform.system()} {platform.release()}"
    info["cpu_count"] = multiprocessing.cpu_count()
    info["cpu_physical"] = psutil.cpu_count(logical=False) if psutil else None
    if psutil:
        try:
            mem_info = psutil.virtual_memory()
            info["memory_mb"] = int(mem_info.total / (1024**2))
        except Exception:
            info["memory_mb"] = None
    else:
        info["memory_mb"] = None
    return _store("sysinfo", info)


# --------------------------------------------------------------------------- #
#   RAG capability detection
# --------------------------------------------------------------------------- #

def detect_rag_capabilities() -> Dict[str, bool]:
    """
    Check for availability of RAG-related libraries and capabilities.
    
    Returns
    -------
    Dict[str, bool]
        Mapping capability name → availability status.
    """
    cached = _cached("rag_caps")
    if cached is not None:
        return cached
    capabilities = {}
    
    # Check for sentence-transformers
    try:
        import sentence_transformers
        capabilities["sentence_transformers"] = True
    except ImportError:
        capabilities["sentence_transformers"] = False
    
    # Check for FAISS
    try:
        import faiss
        capabilities["faiss"] = True
    except ImportError:
        capabilities["faiss"] = False
    
    # Check for LanceDB
    try:
        import lancedb
        capabilities["lancedb"] = True
    except ImportError:
        capabilities["lancedb"] = False
    
    # Check for Ollama availability (for embeddings)
    try:
        import requests
        response = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
        capabilities["ollama"] = response.status_code == 200
    except Exception:
        capabilities["ollama"] = False
    
    return _store("rag_caps", capabilities)


# --------------------------------------------------------------------------- #
#   Public API
# --------------------------------------------------------------------------- #

__all__ = [
    "API_KEY_ENV_VARS",
    "detect_api_keys",
    "list_available_models",
    "detect_system_info",
    "detect_rag_capabilities",
    "refresh_env_detector_cache",
]
