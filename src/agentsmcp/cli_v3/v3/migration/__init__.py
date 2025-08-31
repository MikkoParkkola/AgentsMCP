"""Migration utilities for CLI v3 backward compatibility."""

from .legacy_adapter import LegacyAdapter, LegacyCommandMapping

__all__ = ["LegacyAdapter", "LegacyCommandMapping"]