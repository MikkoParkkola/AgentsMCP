"""Web-related tools for OpenAI Agents SDK integration."""

import json
from typing import Any, Dict, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .base_tools import BaseTool, tool_registry


class WebSearchTool(BaseTool):
    """Tool for performing web searches (simulated - requires actual search API)."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information on a given topic.",
        )

    def execute(self, query: str, max_results: int = 5) -> str:
        """Perform web search (simulated implementation)."""
        try:
            if not query.strip():
                return "Error: Empty search query provided"

            # This is a simulated implementation
            # In production, you would integrate with actual search APIs like:
            # - Google Custom Search API
            # - Bing Search API
            # - DuckDuckGo API
            # - SerpAPI

            self.logger.debug(
                f"Performing web search: {query} (max_results: {max_results})"
            )

            return f"""Web Search Results for "{query}" (SIMULATED):

Note: This is a simulated web search tool. To enable actual web search, integrate with:
- Google Custom Search API
- Bing Search API  
- DuckDuckGo API
- SerpAPI or similar service

Query: {query}
Max results requested: {max_results}

To implement real web search:
1. Get API credentials from your chosen search provider
2. Install required client library (e.g., google-api-python-client)
3. Replace this simulation with actual API calls
4. Handle rate limiting and error responses appropriately

Example integration patterns are available in the tool implementation."""

        except Exception as e:
            self.logger.exception("Error performing web search")
            return f"Error performing web search: {str(e)}"

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query to look for"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        }


class HttpRequestTool(BaseTool):
    """Tool for making HTTP requests to web APIs."""

    def __init__(self):
        super().__init__(
            name="http_request",
            description="Make HTTP requests to web APIs and return the response.",
        )

    def execute(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> str:
        """Make HTTP request."""
        try:
            if not HTTPX_AVAILABLE:
                return "Error: httpx library not available. Install with 'pip install httpx'"

            if not url.strip():
                return "Error: Empty URL provided"

            method = method.upper()
            if method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]:
                return f"Error: Unsupported HTTP method: {method}"

            # Prepare request
            request_data = None
            if data:
                try:
                    # Try to parse as JSON first
                    request_data = json.loads(data)
                except json.JSONDecodeError:
                    # Use as raw string data
                    request_data = data

            self.logger.debug(f"Making {method} request to {url}")

            with httpx.Client(timeout=timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=headers or {},
                    params=params or {},
                    json=request_data if isinstance(request_data, dict) else None,
                    data=request_data if isinstance(request_data, str) else None,
                )

                # Format response
                result = f"""HTTP {method} Request to {url}:

STATUS: {response.status_code} {response.reason_phrase}

HEADERS:
{self._format_headers(dict(response.headers))}

RESPONSE BODY:
{self._format_response_body(response)}"""

                self.logger.debug(f"Request completed: {response.status_code}")
                return result

        except httpx.TimeoutException:
            return f"Error: Request timed out after {timeout} seconds"
        except httpx.RequestError as e:
            return f"Error making request: {str(e)}"
        except Exception as e:
            self.logger.exception("Error making HTTP request")
            return f"Error making HTTP request: {str(e)}"

    def _format_headers(self, headers: Dict[str, str]) -> str:
        """Format headers for display."""
        if not headers:
            return "(No headers)"

        formatted = []
        for key, value in headers.items():
            # Truncate very long header values
            display_value = value if len(value) <= 100 else value[:97] + "..."
            formatted.append(f"  {key}: {display_value}")

        return "\n".join(formatted)

    def _format_response_body(self, response: "httpx.Response") -> str:
        """Format response body for display."""
        try:
            content_type = response.headers.get("content-type", "").lower()

            if "application/json" in content_type:
                # Pretty print JSON
                try:
                    json_data = response.json()
                    return json.dumps(json_data, indent=2)
                except (ValueError, TypeError, AttributeError):
                    return response.text

            elif any(
                t in content_type
                for t in ["text/", "application/xml", "application/javascript"]
            ):
                # Return text content, truncated if too long
                text = response.text
                if len(text) > 2000:
                    return text[:1997] + "..."
                return text

            else:
                # Binary or unknown content type
                return f"({content_type or 'unknown'} content, {len(response.content)} bytes)"

        except Exception:
            return f"(Unable to decode response body, {len(response.content)} bytes)"

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to make the request to",
                    "format": "uri",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to use",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                    "default": "GET",
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP headers to send with the request",
                    "additionalProperties": {"type": "string"},
                },
                "data": {
                    "type": "string",
                    "description": "Request body data (JSON string or raw text)",
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters to add to the URL",
                    "additionalProperties": {"type": "string"},
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds (default: 30)",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 300,
                },
            },
            "required": ["url"],
        }


# Create and register tool instances
web_search_tool = WebSearchTool()
http_request_tool = HttpRequestTool()

tool_registry.register(web_search_tool)
tool_registry.register(http_request_tool)
