"""Tests for PII sanitization pipeline.

This module tests the PII sanitization system including:
- Detection of various PII types
- Different sanitization levels
- Performance and accuracy validation
- Custom rule support
"""

import pytest
import re
from unittest.mock import Mock

from ..pii_sanitizer import PIISanitizer, SanitizationRule, PIIType, SanitizationStats
from ..log_schemas import SanitizationLevel, UserInteractionEvent


class TestPIISanitizer:
    """Test PII sanitization functionality."""
    
    def test_initialization(self):
        """Test sanitizer initialization with different levels."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        assert sanitizer.level == SanitizationLevel.STANDARD
        assert len(sanitizer._active_rules) > 0
        assert isinstance(sanitizer.stats, SanitizationStats)
    
    def test_email_sanitization(self):
        """Test email address sanitization."""
        sanitizer = PIISanitizer(level=SanitizationLevel.MINIMAL)
        
        test_text = "Contact me at john.doe@example.com or jane@company.org"
        sanitized = sanitizer._sanitize_string(test_text)
        
        assert "john.doe@example.com" not in sanitized
        assert "jane@company.org" not in sanitized
        assert "[EMAIL_REDACTED]" in sanitized
    
    def test_phone_number_sanitization(self):
        """Test phone number sanitization."""
        sanitizer = PIISanitizer(level=SanitizationLevel.MINIMAL)
        
        test_cases = [
            "Call me at (555) 123-4567",
            "Phone: 555-123-4567", 
            "Contact: +1-555-123-4567",
            "My number is 5551234567"
        ]
        
        for test_text in test_cases:
            sanitized = sanitizer._sanitize_string(test_text)
            assert "[PHONE_REDACTED]" in sanitized
            
            # Verify no phone numbers remain
            phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b')
            assert not phone_pattern.search(sanitized)
    
    def test_ssn_sanitization(self):
        """Test Social Security Number sanitization."""
        sanitizer = PIISanitizer(level=SanitizationLevel.MINIMAL)
        
        test_cases = [
            "SSN: 123-45-6789",
            "Social Security: 123.45.6789",
            "My SSN is 123456789"
        ]
        
        for test_text in test_cases:
            sanitized = sanitizer._sanitize_string(test_text)
            assert "[SSN_REDACTED]" in sanitized
            assert "123-45-6789" not in sanitized
            assert "123456789" not in sanitized
    
    def test_credit_card_sanitization(self):
        """Test credit card number sanitization."""
        sanitizer = PIISanitizer(level=SanitizationLevel.MINIMAL)
        
        test_cases = [
            "Visa: 4111111111111111",
            "MasterCard: 5555555555554444",
            "Amex: 378282246310005"
        ]
        
        for test_text in test_cases:
            sanitized = sanitizer._sanitize_string(test_text)
            assert "[CARD_REDACTED]" in sanitized
            # Verify no card numbers remain
            for case in test_cases:
                number = case.split(': ')[1]
                assert number not in sanitized
    
    def test_sanitization_levels(self):
        """Test different sanitization levels."""
        test_text = "Contact John Doe at john@example.com, IP: 192.168.1.1, Amount: $1,500.00"
        
        # Minimal level - should only catch basic PII
        minimal = PIISanitizer(level=SanitizationLevel.MINIMAL)
        minimal_result = minimal._sanitize_string(test_text)
        
        assert "[EMAIL_REDACTED]" in minimal_result
        assert "John Doe" in minimal_result  # Names not redacted at minimal level
        assert "192.168.1.1" in minimal_result  # IPs not redacted at minimal level
        
        # Standard level - should catch more PII
        standard = PIISanitizer(level=SanitizationLevel.STANDARD)
        standard_result = standard._sanitize_string(test_text)
        
        assert "[EMAIL_REDACTED]" in standard_result
        assert "[IP_REDACTED]" in standard_result
        assert "192.168.1.1" not in standard_result
        
        # Strict level - should be most aggressive
        strict = PIISanitizer(level=SanitizationLevel.STRICT)
        strict_result = strict._sanitize_string(test_text)
        
        assert "[EMAIL_REDACTED]" in strict_result
        assert "[IP_REDACTED]" in strict_result
        assert "[AMOUNT_REDACTED]" in strict_result
        assert "$1,500.00" not in strict_result
    
    def test_event_sanitization(self):
        """Test sanitization of event objects."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        event = UserInteractionEvent(
            user_input="My email is user@test.com and phone is 555-1234",
            assistant_response="I'll contact you at user@test.com",
            session_id="session_123"
        )
        
        sanitized_event = sanitizer.sanitize_event(event)
        
        assert "[EMAIL_REDACTED]" in sanitized_event.user_input
        assert "[PHONE_REDACTED]" in sanitized_event.user_input
        assert "[EMAIL_REDACTED]" in sanitized_event.assistant_response
        assert "user@test.com" not in sanitized_event.user_input
        assert "user@test.com" not in sanitized_event.assistant_response
    
    def test_batch_sanitization(self):
        """Test batch sanitization of multiple events."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        events = [
            UserInteractionEvent(
                user_input=f"Test {i} with email test{i}@example.com",
                assistant_response=f"Response {i}",
                session_id=f"session_{i}"
            )
            for i in range(5)
        ]
        
        sanitized_events = sanitizer.sanitize_batch(events)
        
        assert len(sanitized_events) == 5
        
        for event in sanitized_events:
            assert "[EMAIL_REDACTED]" in event.user_input
            assert "@example.com" not in event.user_input
    
    def test_custom_rules(self):
        """Test adding custom sanitization rules."""
        # Custom rule to redact API keys
        api_key_rule = SanitizationRule(
            pattern=re.compile(r'\bapi_key_[a-zA-Z0-9]{32}\b'),
            replacement="[API_KEY_REDACTED]",
            description="API keys",
            level=SanitizationLevel.MINIMAL
        )
        
        sanitizer = PIISanitizer(
            level=SanitizationLevel.MINIMAL,
            custom_rules=[api_key_rule]
        )
        
        test_text = "Using API key: api_key_abc123def456ghi789jkl012mno345pq"
        sanitized = sanitizer._sanitize_string(test_text)
        
        assert "[API_KEY_REDACTED]" in sanitized
        assert "api_key_abc123def456ghi789jkl012mno345pq" not in sanitized
    
    def test_statistics_tracking(self):
        """Test sanitization statistics collection."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        # Process some events with PII
        test_events = [
            UserInteractionEvent(
                user_input="Email: user1@test.com, Phone: 555-0001",
                assistant_response="Thanks!",
                session_id="stats_test_1"
            ),
            UserInteractionEvent(
                user_input="Contact: user2@test.com",
                assistant_response="Got it!",
                session_id="stats_test_2"
            )
        ]
        
        sanitizer.sanitize_batch(test_events)
        
        stats = sanitizer.get_stats()
        
        assert stats.total_events_processed >= 2
        assert stats.events_modified >= 2  # Should have modified events with PII
        assert stats.sanitization_time_ms > 0
        assert len(stats.pii_instances_found) > 0
    
    def test_hash_preserving_sanitizer(self):
        """Test hash-preserving sanitization for analytics."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        hash_sanitizer = sanitizer.create_hash_preserving_sanitizer(salt="test_salt")
        
        test_text = "Contact john@example.com for more info"
        sanitized1 = hash_sanitizer(test_text)
        sanitized2 = hash_sanitizer(test_text)
        
        # Same input should produce same hash
        assert sanitized1 == sanitized2
        
        # Should contain hashed replacement
        assert "EMAIL_HASH_" in sanitized1
        assert "john@example.com" not in sanitized1
    
    def test_validation_functionality(self):
        """Test sanitization validation."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        original = "Contact me at john.doe@example.com or call 555-1234"
        sanitized = sanitizer._sanitize_string(original)
        
        validation = sanitizer.validate_sanitization(original, sanitized)
        
        assert len(validation['pii_found']) > 0
        assert validation['sanitization_effective'] == True
        assert validation['preservation_score'] > 0
        assert validation['preservation_score'] <= 1.0
    
    def test_nested_dict_sanitization(self):
        """Test sanitization of nested dictionary structures."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        test_data = {
            'user_info': {
                'email': 'user@test.com',
                'contact': {
                    'phone': '555-1234',
                    'address': '123 Main St'
                }
            },
            'metadata': {
                'ip_address': '192.168.1.1',
                'api_key': 'sk_test_1234567890abcdef'
            }
        }
        
        sanitized = sanitizer._sanitize_dict(test_data)
        
        # Check nested sanitization
        assert "[EMAIL_REDACTED]" in sanitized['user_info']['email']
        assert "[PHONE_REDACTED]" in sanitized['user_info']['contact']['phone']
        assert "[IP_REDACTED]" in sanitized['metadata']['ip_address']
        
        # Verify original PII is removed
        assert "user@test.com" not in str(sanitized)
        assert "555-1234" not in str(sanitized)
        assert "192.168.1.1" not in str(sanitized)
    
    def test_performance_with_large_text(self):
        """Test sanitization performance with large text inputs."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        # Create large text with scattered PII
        large_text = ""
        for i in range(1000):
            if i % 10 == 0:
                large_text += f"Contact user{i}@example.com or call 555-000{i % 10}. "
            else:
                large_text += f"This is some regular text content {i}. "
        
        import time
        start_time = time.perf_counter()
        
        sanitized = sanitizer._sanitize_string(large_text)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Should complete reasonably quickly (less than 100ms for this size)
        assert elapsed_ms < 100
        
        # Verify sanitization worked
        assert "[EMAIL_REDACTED]" in sanitized
        assert "[PHONE_REDACTED]" in sanitized
        assert "@example.com" not in sanitized
    
    def test_no_sanitization_level(self):
        """Test NONE sanitization level preserves all data."""
        sanitizer = PIISanitizer(level=SanitizationLevel.NONE)
        
        test_text = "Email: user@test.com, Phone: 555-1234, SSN: 123-45-6789"
        sanitized = sanitizer._sanitize_string(test_text)
        
        # Should be unchanged
        assert sanitized == test_text
        assert "user@test.com" in sanitized
        assert "555-1234" in sanitized
        assert "123-45-6789" in sanitized
    
    def test_error_handling(self):
        """Test error handling in sanitization process."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        # Create a mock event that will cause serialization error
        mock_event = Mock()
        mock_event.to_dict.side_effect = Exception("Serialization error")
        
        # Should not raise exception, should return original event
        result = sanitizer.sanitize_event(mock_event)
        assert result == mock_event
        
        # Should track the error
        stats = sanitizer.get_stats()
        assert len(stats.errors) > 0
    
    def test_rule_level_filtering(self):
        """Test that rules are filtered by sanitization level."""
        # Create sanitizer with minimal level
        minimal_sanitizer = PIISanitizer(level=SanitizationLevel.MINIMAL)
        minimal_rules = len(minimal_sanitizer._active_rules)
        
        # Create sanitizer with paranoid level
        paranoid_sanitizer = PIISanitizer(level=SanitizationLevel.PARANOID)
        paranoid_rules = len(paranoid_sanitizer._active_rules)
        
        # Paranoid should have more active rules than minimal
        assert paranoid_rules >= minimal_rules
    
    def test_pii_type_inference(self):
        """Test PII type inference from rule descriptions."""
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        # Test a few key PII type inferences
        test_cases = [
            ("Email addresses", PIIType.EMAIL),
            ("Phone numbers", PIIType.PHONE),
            ("Social Security Numbers", PIIType.SSN),
            ("Credit card numbers", PIIType.CREDIT_CARD),
            ("IP addresses", PIIType.IP_ADDRESS),
            ("Custom pattern", PIIType.CUSTOM)
        ]
        
        for description, expected_type in test_cases:
            mock_rule = Mock()
            mock_rule.description = description
            
            inferred_type = sanitizer._infer_pii_type_from_rule(mock_rule)
            assert inferred_type == expected_type