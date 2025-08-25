import pytest
import requests
import time

@pytest.mark.api
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_health_endpoint(web_server, api_endpoint):
    """Test that the /health endpoint returns proper status."""
    response = requests.get(f"{api_endpoint}/health", timeout=5)
    assert response.status_code == 200
    
    # Try to parse JSON response
    try:
        data = response.json()
        # Should contain status information
        assert "status" in data or "health" in data or len(data) > 0
    except ValueError:
        # Some implementations might return plain text
        assert len(response.text) > 0

@pytest.mark.api
@pytest.mark.flaky(reruns=3, reruns_delay=2) 
def test_docs_endpoint(web_server, api_endpoint):
    """Test that the /docs endpoint serves API documentation."""
    response = requests.get(f"{api_endpoint}/docs", timeout=5)
    assert response.status_code == 200
    
    # Should be HTML content with API docs
    content_type = response.headers.get('content-type', '')
    assert 'html' in content_type.lower() or 'text' in content_type.lower()
    
    # Should contain some indication it's API documentation
    content = response.text.lower()
    assert any(keyword in content for keyword in ['swagger', 'openapi', 'api', 'docs', 'documentation'])

@pytest.mark.api
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_jobs_endpoint(web_server, api_endpoint):
    """Test that the /jobs endpoint returns job information."""
    response = requests.get(f"{api_endpoint}/jobs", timeout=5)
    assert response.status_code == 200
    
    # Try to parse JSON response
    try:
        data = response.json()
        # Should be a list or dict containing job information
        assert isinstance(data, (list, dict))
    except ValueError:
        # Some implementations might return plain text
        assert len(response.text) > 0

@pytest.mark.api
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_api_endpoints_accessibility(web_server, api_endpoint):
    """Test that all documented API endpoints are accessible."""
    endpoints = ["/health", "/docs", "/jobs"]
    
    for endpoint in endpoints:
        response = requests.get(f"{api_endpoint}{endpoint}", timeout=5)
        
        # Should not return server errors (5xx)
        assert response.status_code < 500, f"Server error on {endpoint}: {response.status_code}"
        
        # Should return some content
        assert len(response.text) > 0, f"Empty response from {endpoint}"

@pytest.mark.api
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_api_cors_headers(web_server, api_endpoint):
    """Test that API includes proper CORS headers if needed."""
    response = requests.get(f"{api_endpoint}/health", timeout=5)
    assert response.status_code == 200
    
    # CORS headers are optional but if present should be valid
    if 'access-control-allow-origin' in response.headers:
        cors_origin = response.headers['access-control-allow-origin']
        assert cors_origin in ['*', 'localhost', '127.0.0.1']

@pytest.mark.api
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_api_response_times(web_server, api_endpoint):
    """Test that API endpoints respond within reasonable time."""
    endpoints = ["/health", "/docs", "/jobs"]
    
    for endpoint in endpoints:
        start_time = time.time()
        response = requests.get(f"{api_endpoint}{endpoint}", timeout=5)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within 3 seconds
        assert response_time < 3.0, f"{endpoint} took {response_time:.2f}s to respond"
        
        # Should not return server errors
        assert response.status_code < 500