"""Unit tests for test_function module."""

import unittest
from .test_function import calculate_improvement_score, format_test_result


class TestCalculateImprovementScore(unittest.TestCase):
    """Test cases for calculate_improvement_score function."""
    
    def test_empty_metrics(self):
        """Test with empty metrics dictionary."""
        self.assertEqual(calculate_improvement_score({}), 0.0)
    
    def test_good_metrics(self):
        """Test with good performance metrics."""
        metrics = {
            'success_rate': 0.9,
            'response_time': 500,
            'error_rate': 0.05
        }
        score = calculate_improvement_score(metrics)
        self.assertGreater(score, 0.8)
    
    def test_poor_metrics(self):
        """Test with poor performance metrics."""
        metrics = {
            'success_rate': 0.5,
            'response_time': 2000,
            'error_rate': 0.3
        }
        score = calculate_improvement_score(metrics)
        self.assertLess(score, 0.5)


class TestFormatTestResult(unittest.TestCase):
    """Test cases for format_test_result function."""
    
    def test_passing_test(self):
        """Test formatting for passing test."""
        result = format_test_result("test_example", True, "All assertions passed")
        self.assertIn("✅ PASS", result)
        self.assertIn("test_example", result)
    
    def test_failing_test(self):
        """Test formatting for failing test."""
        result = format_test_result("test_example", False, "Assertion failed")
        self.assertIn("❌ FAIL", result)
        self.assertIn("test_example", result)


if __name__ == '__main__':
    unittest.main()
