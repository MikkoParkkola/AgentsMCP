"""Test utility function for end-to-end testing."""

def calculate_improvement_score(metrics: dict) -> float:
    """Calculate a simple improvement score based on metrics.
    
    Args:
        metrics: Dictionary containing performance metrics
        
    Returns:
        float: Improvement score between 0.0 and 1.0
    """
    if not metrics:
        return 0.0
    
    # Simple scoring algorithm for demonstration
    score = 0.0
    if metrics.get('success_rate', 0) > 0.8:
        score += 0.4
    if metrics.get('response_time', float('inf')) < 1000:
        score += 0.3
    if metrics.get('error_rate', 1.0) < 0.1:
        score += 0.3
    
    return min(score, 1.0)


def format_test_result(test_name: str, passed: bool, details: str = "") -> str:
    """Format test result for display.
    
    Args:
        test_name: Name of the test
        passed: Whether the test passed
        details: Additional details about the test
        
    Returns:
        str: Formatted test result string
    """
    status = "✅ PASS" if passed else "❌ FAIL"
    result = f"{status}: {test_name}"
    if details:
        result += f" - {details}"
    return result
