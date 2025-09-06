"""
Tests for validation helper utilities.

Part of end-to-end self-improvement test.
"""

import pytest
from agentsmcp.utils.validation_helpers import (
    validate_non_empty_string,
    validate_positive_number,
    validate_email_format
)


def test_validate_non_empty_string():
    """Test string validation."""
    # Valid cases
    assert validate_non_empty_string("hello") == "hello"
    assert validate_non_empty_string("  world  ") == "world"
    
    # Invalid cases
    with pytest.raises(ValueError, match="must be a string"):
        validate_non_empty_string(123)
    
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_non_empty_string("")
    
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_non_empty_string("   ")


def test_validate_positive_number():
    """Test positive number validation."""
    # Valid cases
    assert validate_positive_number(5) == 5.0
    assert validate_positive_number("3.14") == 3.14
    assert validate_positive_number(1.5) == 1.5
    
    # Invalid cases
    with pytest.raises(ValueError, match="must be a number"):
        validate_positive_number("not_a_number")
    
    with pytest.raises(ValueError, match="must be positive"):
        validate_positive_number(0)
    
    with pytest.raises(ValueError, match="must be positive"):
        validate_positive_number(-5)


def test_validate_email_format():
    """Test email format validation."""
    # Valid cases
    assert validate_email_format("user@example.com") == "user@example.com"
    assert validate_email_format("TEST@DOMAIN.ORG") == "test@domain.org"
    
    # Invalid cases
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_email_format("")
    
    with pytest.raises(ValueError, match="invalid format"):
        validate_email_format("notanemail")
    
    with pytest.raises(ValueError, match="invalid format"):
        validate_email_format("user@")
    
    with pytest.raises(ValueError, match="invalid format"):
        validate_email_format("@example.com")
