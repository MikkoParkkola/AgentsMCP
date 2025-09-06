"""
Validation helper utilities for AgentsMCP.

Added during end-to-end self-improvement test to demonstrate
autonomous code improvement capability.
"""

def validate_non_empty_string(value, field_name="field"):
    """
    Validate that a value is a non-empty string.
    
    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        
    Returns:
        str: The validated string value
        
    Raises:
        ValueError: If value is empty or not a string
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
    
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    
    return value.strip()


def validate_positive_number(value, field_name="field"):
    """
    Validate that a value is a positive number.
    
    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        
    Returns:
        float: The validated number
        
    Raises:
        ValueError: If value is not a positive number
    """
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a number, got {value}")
    
    if num_value <= 0:
        raise ValueError(f"{field_name} must be positive, got {num_value}")
    
    return num_value


def validate_email_format(email, field_name="email"):
    """
    Basic email format validation.
    
    Args:
        email: Email string to validate
        field_name: Name of the field for error messages
        
    Returns:
        str: The validated email
        
    Raises:
        ValueError: If email format is invalid
    """
    email = validate_non_empty_string(email, field_name)
    
    # Check basic email structure: must have @ with content before and after
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError(f"{field_name} has invalid format: {email}")
    
    # Check that domain part has a dot
    domain_part = email.split("@")[-1]
    if "." not in domain_part:
        raise ValueError(f"{field_name} has invalid format: {email}")
    
    return email.lower()
