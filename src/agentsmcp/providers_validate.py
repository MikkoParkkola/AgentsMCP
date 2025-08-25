"""Provider configuration validation helpers (K1).

This module provides non-raising validation helpers for provider configs.
Each function returns a ValidationResult with ok/reason/details instead of
raising exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

from .config import ProviderType, ProviderConfig


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""
    details: Optional[str] = None


def validate_provider_config(provider: ProviderType, config: ProviderConfig) -> ValidationResult:
    """Validate provider config by probing a lightweight endpoint.

    - Never raises; returns ValidationResult
    - Tries a GET on a minimal endpoint per provider
    - Only validates reachability/auth, not model fitness
    """
    try:
        if provider == ProviderType.OPENAI:
            if not config.api_key:
                return ValidationResult(False, "missing_api_key", "Set OPENAI_API_KEY or configure providers.openai.api_key")
            base = (config.api_base.rstrip("/")) if config.api_base else "https://api.openai.com/v1"
            url = f"{base}/models"
            headers = {"Authorization": f"Bearer {config.api_key}", "Accept": "application/json"}
            with httpx.Client(timeout=10) as client:
                r = client.get(url, headers=headers)
            if r.status_code in (401, 403):
                return ValidationResult(False, "auth_failed", f"HTTP {r.status_code}")
            if r.status_code >= 500:
                return ValidationResult(False, "provider_unavailable", f"HTTP {r.status_code}")
            if r.status_code >= 400:
                return ValidationResult(False, "protocol_error", f"HTTP {r.status_code}")
            return ValidationResult(True, "ok")

        if provider == ProviderType.OPENROUTER:
            if not config.api_key:
                return ValidationResult(False, "missing_api_key", "Set OPENROUTER_API_KEY or configure providers.openrouter.api_key")
            base = (config.api_base.rstrip("/")) if config.api_base else "https://openrouter.ai/api/v1"
            url = f"{base}/models"
            headers = {"Authorization": f"Bearer {config.api_key}", "Accept": "application/json"}
            with httpx.Client(timeout=10) as client:
                r = client.get(url, headers=headers)
            if r.status_code in (401, 403):
                return ValidationResult(False, "auth_failed", f"HTTP {r.status_code}")
            if r.status_code >= 500:
                return ValidationResult(False, "provider_unavailable", f"HTTP {r.status_code}")
            if r.status_code >= 400:
                return ValidationResult(False, "protocol_error", f"HTTP {r.status_code}")
            return ValidationResult(True, "ok")

        if provider == ProviderType.OLLAMA:
            base = (config.api_base.rstrip("/")) if config.api_base else "http://localhost:11434"
            url = f"{base}/api/tags"
            with httpx.Client(timeout=5) as client:
                r = client.get(url, headers={"Accept": "application/json"})
            if r.status_code >= 500:
                return ValidationResult(False, "provider_unavailable", f"HTTP {r.status_code}")
            if r.status_code >= 400:
                return ValidationResult(False, "protocol_error", f"HTTP {r.status_code}")
            return ValidationResult(True, "ok")

        # CUSTOM: cannot validate without a known scheme; treat as okay if api_base is set
        return ValidationResult(bool(config.api_base), "ok" if config.api_base else "missing_api_base")

    except (httpx.ConnectError, httpx.ReadTimeout):
        return ValidationResult(False, "network_error", "Could not reach provider endpoint")
    except Exception as e:  # pragma: no cover - unexpected
        return ValidationResult(False, "unexpected_error", str(e))


__all__ = ["ValidationResult", "validate_provider_config"]

