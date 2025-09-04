"""PII Sanitization pipeline for agent execution logs.

This module provides comprehensive PII sanitization capabilities to ensure
privacy compliance while preserving data utility for retrospective analysis.
Supports multiple sanitization levels from minimal to paranoid.
"""

import re
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Set, Callable, Pattern
from dataclasses import dataclass, field
from enum import Enum

from .log_schemas import SanitizationLevel, AgentEvent

logger = logging.getLogger(__name__)


@dataclass
class SanitizationRule:
    """Defines a rule for sanitizing specific types of PII."""
    pattern: Pattern[str]
    replacement: str
    description: str
    level: SanitizationLevel
    preserve_structure: bool = True


class PIIType(Enum):
    """Types of PII that can be detected and sanitized."""
    EMAIL = "email"
    PHONE = "phone" 
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    TOKEN = "token"
    PASSWORD = "password"
    NAME = "name"
    ADDRESS = "address"
    FINANCIAL = "financial"
    CUSTOM = "custom"


@dataclass
class SanitizationStats:
    """Statistics about sanitization operations."""
    total_events_processed: int = 0
    events_modified: int = 0
    pii_instances_found: Dict[PIIType, int] = field(default_factory=dict)
    sanitization_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class PIISanitizer:
    """High-performance PII sanitization system for agent execution logs.
    
    Provides configurable sanitization levels and maintains statistical tracking
    for compliance reporting and performance monitoring.
    """
    
    def __init__(
        self,
        level: SanitizationLevel = SanitizationLevel.STANDARD,
        custom_rules: Optional[List[SanitizationRule]] = None,
        preserve_analytics: bool = True
    ):
        """Initialize the PII sanitizer.
        
        Args:
            level: Sanitization level to apply
            custom_rules: Additional custom sanitization rules
            preserve_analytics: Whether to preserve data useful for analytics
        """
        self.level = level
        self.preserve_analytics = preserve_analytics
        self.stats = SanitizationStats()
        
        # Initialize built-in rules
        self._rules: List[SanitizationRule] = []
        self._init_builtin_rules()
        
        # Add custom rules
        if custom_rules:
            self._rules.extend(custom_rules)
        
        # Filter rules by current sanitization level
        self._active_rules = [
            rule for rule in self._rules 
            if self._should_apply_rule(rule, level)
        ]
        
        logger.info(
            f"PIISanitizer initialized with {len(self._active_rules)} active rules "
            f"at level {level.value}"
        )
    
    def _init_builtin_rules(self) -> None:
        """Initialize built-in PII sanitization rules."""
        
        # Email addresses - all levels
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            replacement="[EMAIL_REDACTED]",
            description="Email addresses",
            level=SanitizationLevel.MINIMAL
        ))
        
        # Phone numbers - all levels
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            replacement="[PHONE_REDACTED]",
            description="Phone numbers",
            level=SanitizationLevel.MINIMAL
        ))
        
        # Social Security Numbers - all levels
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'\b(?!000|666|9\d{2})\d{3}[-.]?(?!00)\d{2}[-.]?(?!0000)\d{4}\b'),
            replacement="[SSN_REDACTED]",
            description="Social Security Numbers",
            level=SanitizationLevel.MINIMAL
        ))
        
        # Credit card numbers - all levels
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            replacement="[CARD_REDACTED]",
            description="Credit card numbers",
            level=SanitizationLevel.MINIMAL
        ))
        
        # IP addresses - standard and above
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            replacement="[IP_REDACTED]",
            description="IP addresses",
            level=SanitizationLevel.STANDARD
        ))
        
        # API keys and tokens - standard and above
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'\b[A-Za-z0-9]{20,}\b'),
            replacement="[TOKEN_REDACTED]",
            description="Potential API keys/tokens",
            level=SanitizationLevel.STANDARD
        ))
        
        # Common password patterns - standard and above
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'(?i)password["\s]*[:=]["\s]*[^\s"]+'),
            replacement='password: "[REDACTED]"',
            description="Password fields",
            level=SanitizationLevel.STANDARD
        ))
        
        # Financial information - strict and above
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'\b\$[0-9,]+(?:\.[0-9]{2})?\b'),
            replacement="[AMOUNT_REDACTED]",
            description="Dollar amounts",
            level=SanitizationLevel.STRICT
        ))
        
        # Names (aggressive) - strict and above
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
            replacement="[NAME_REDACTED]",
            description="Potential person names",
            level=SanitizationLevel.STRICT
        ))
        
        # Paranoid: Any sequences that look like identifiers
        self._rules.append(SanitizationRule(
            pattern=re.compile(r'\b[A-Za-z0-9]{8,}\b'),
            replacement="[ID_REDACTED]",
            description="Potential identifiers",
            level=SanitizationLevel.PARANOID
        ))
    
    def _should_apply_rule(self, rule: SanitizationRule, level: SanitizationLevel) -> bool:
        """Determine if a rule should be applied at the given sanitization level."""
        level_order = [
            SanitizationLevel.NONE,
            SanitizationLevel.MINIMAL,
            SanitizationLevel.STANDARD,
            SanitizationLevel.STRICT,
            SanitizationLevel.PARANOID
        ]
        
        return level_order.index(level) >= level_order.index(rule.level)
    
    def sanitize_event(self, event: AgentEvent) -> AgentEvent:
        """Sanitize a single agent event.
        
        Args:
            event: The event to sanitize
            
        Returns:
            Sanitized event with PII removed or masked
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Convert event to dict for easier processing
            event_dict = event.to_dict()
            original_dict = event_dict.copy()
            
            # Sanitize the dictionary
            sanitized_dict = self._sanitize_dict(event_dict)
            
            # Update statistics
            self.stats.total_events_processed += 1
            if sanitized_dict != original_dict:
                self.stats.events_modified += 1
            
            # Convert back to event object
            # Note: This is simplified - in practice you'd need proper deserialization
            for key, value in sanitized_dict.items():
                if hasattr(event, key):
                    setattr(event, key, value)
            
            return event
            
        except Exception as e:
            error_msg = f"Error sanitizing event {getattr(event, 'event_id', 'unknown')}: {e}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return event
        
        finally:
            # Track sanitization time
            self.stats.sanitization_time_ms += (time.perf_counter() - start_time) * 1000
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize a dictionary structure."""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self._sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_dict(item) if isinstance(item, dict)
                    else self._sanitize_string(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_string(self, text: str) -> str:
        """Apply sanitization rules to a text string."""
        if not text or self.level == SanitizationLevel.NONE:
            return text
        
        sanitized = text
        
        for rule in self._active_rules:
            original = sanitized
            sanitized = rule.pattern.sub(rule.replacement, sanitized)
            
            # Track PII instances found (approximate)
            if original != sanitized:
                pii_type = self._infer_pii_type_from_rule(rule)
                if pii_type:
                    count = len(rule.pattern.findall(original))
                    self.stats.pii_instances_found[pii_type] = (
                        self.stats.pii_instances_found.get(pii_type, 0) + count
                    )
        
        return sanitized
    
    def _infer_pii_type_from_rule(self, rule: SanitizationRule) -> Optional[PIIType]:
        """Infer PII type from rule description."""
        description_lower = rule.description.lower()
        
        if "email" in description_lower:
            return PIIType.EMAIL
        elif "phone" in description_lower:
            return PIIType.PHONE
        elif "ssn" in description_lower:
            return PIIType.SSN
        elif "card" in description_lower or "credit" in description_lower:
            return PIIType.CREDIT_CARD
        elif "ip" in description_lower:
            return PIIType.IP_ADDRESS
        elif "token" in description_lower or "api" in description_lower:
            return PIIType.API_KEY
        elif "password" in description_lower:
            return PIIType.PASSWORD
        elif "name" in description_lower:
            return PIIType.NAME
        elif "amount" in description_lower or "financial" in description_lower:
            return PIIType.FINANCIAL
        else:
            return PIIType.CUSTOM
    
    def sanitize_batch(self, events: List[AgentEvent]) -> List[AgentEvent]:
        """Sanitize a batch of events efficiently.
        
        Args:
            events: List of events to sanitize
            
        Returns:
            List of sanitized events
        """
        return [self.sanitize_event(event) for event in events]
    
    def get_stats(self) -> SanitizationStats:
        """Get sanitization statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset sanitization statistics."""
        self.stats = SanitizationStats()
    
    def add_custom_rule(self, rule: SanitizationRule) -> None:
        """Add a custom sanitization rule."""
        self._rules.append(rule)
        
        # Update active rules if this rule should be applied at current level
        if self._should_apply_rule(rule, self.level):
            self._active_rules.append(rule)
            logger.info(f"Added custom rule: {rule.description}")
    
    def create_hash_preserving_sanitizer(self, salt: str = "") -> Callable[[str], str]:
        """Create a sanitizer that replaces PII with consistent hashes.
        
        This allows for some analytics while preserving privacy by ensuring
        the same PII always maps to the same hash.
        
        Args:
            salt: Salt to add to hashing for additional security
            
        Returns:
            Function that sanitizes strings using consistent hashing
        """
        def hash_sanitizer(text: str) -> str:
            if not text:
                return text
            
            sanitized = text
            
            for rule in self._active_rules:
                def hash_replacement(match):
                    # Create consistent hash for this PII instance
                    pii_value = match.group(0)
                    hash_input = f"{salt}{pii_value}".encode('utf-8')
                    hash_digest = hashlib.sha256(hash_input).hexdigest()[:8]
                    return f"[{rule.description.upper()}_HASH_{hash_digest}]"
                
                sanitized = rule.pattern.sub(hash_replacement, sanitized)
            
            return sanitized
        
        return hash_sanitizer
    
    def validate_sanitization(self, original: str, sanitized: str) -> Dict[str, Any]:
        """Validate that sanitization was effective.
        
        Args:
            original: Original text
            sanitized: Sanitized text
            
        Returns:
            Validation results with found PII and sanitization effectiveness
        """
        results = {
            'pii_found': [],
            'potentially_missed': [],
            'sanitization_effective': True,
            'preservation_score': 0.0
        }
        
        # Check each rule against original text
        for rule in self._active_rules:
            matches = rule.pattern.findall(original)
            if matches:
                results['pii_found'].append({
                    'type': rule.description,
                    'count': len(matches),
                    'rule_level': rule.level.value
                })
                
                # Check if any matches still exist in sanitized version
                remaining_matches = rule.pattern.findall(sanitized)
                if remaining_matches:
                    results['potentially_missed'].append({
                        'type': rule.description,
                        'remaining_count': len(remaining_matches)
                    })
                    results['sanitization_effective'] = False
        
        # Calculate preservation score (how much useful structure remains)
        if len(original) > 0:
            # Simple metric: ratio of non-redacted characters
            redacted_chars = sanitized.count('[') + sanitized.count(']') + sanitized.count('_REDACTED')
            useful_chars = len(sanitized) - redacted_chars
            results['preservation_score'] = min(1.0, useful_chars / len(original))
        
        return results