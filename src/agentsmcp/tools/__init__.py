"""Tools package for AgentsMCP with OpenAI Agents SDK integration."""

# Import lazy loading utilities
from ..lazy_loading import lazy_import, LazyRegistry

# Lazy imports for tool modules to avoid loading heavy dependencies at startup
analysis_tools = lazy_import('.analysis_tools', __package__)
base_tools = lazy_import('.base_tools', __package__)  
file_tools = lazy_import('.file_tools', __package__)
web_tools = lazy_import('.web_tools', __package__)
mcp_tool = lazy_import('.mcp_tool', __package__)
shell_tools = lazy_import('.shell_tools', __package__)

# Create a lazy tool registry
_lazy_tool_registry = LazyRegistry()

def _register_default_tools():
    """Register default tools in the lazy registry."""
    _lazy_tool_registry.register("file_operations", file_tools.FileOperationTool)
    _lazy_tool_registry.register("text_analysis", analysis_tools.TextAnalysisTool)
    _lazy_tool_registry.register("code_analysis", analysis_tools.CodeAnalysisTool)
    _lazy_tool_registry.register("web_search", web_tools.WebSearchTool)
    _lazy_tool_registry.register("http_request", web_tools.HttpRequestTool)
    _lazy_tool_registry.register("mcp_call", mcp_tool.MCPCallTool)
    _lazy_tool_registry.register("shell_command", shell_tools.ShellCommandTool)

# Initialize the registry on first access
_tools_registered = False

def get_tool_registry():
    """Get the tool registry, initializing it lazily if needed."""
    global _tools_registered
    if not _tools_registered:
        _register_default_tools()
        _tools_registered = True
    return _lazy_tool_registry

# Lazy accessors for tools
def get_base_tool():
    return base_tools.BaseTool

def get_tool_registry_instance():
    return base_tools.tool_registry

# Lazy accessors for specific tools
def get_file_operation_tool():
    return file_tools.FileOperationTool

def get_read_file_tool():
    return file_tools.read_file_tool

def get_write_file_tool():
    return file_tools.write_file_tool

def get_list_directory_tool():
    return file_tools.list_directory_tool

def get_text_analysis_tool():
    return analysis_tools.TextAnalysisTool

def get_code_analysis_tool():
    return analysis_tools.CodeAnalysisTool

def get_text_analysis_tool_instance():
    return analysis_tools.text_analysis_tool

def get_code_analysis_tool_instance():
    return analysis_tools.code_analysis_tool

def get_web_search_tool():
    return web_tools.WebSearchTool

def get_http_request_tool():
    return web_tools.HttpRequestTool

def get_web_search_tool_instance():
    return web_tools.web_search_tool

def get_http_request_tool_instance():
    return web_tools.http_request_tool

def get_mcp_call_tool():
    return mcp_tool.MCPCallTool

def get_shell_command_tool():
    return shell_tools.ShellCommandTool

def get_run_shell_tool():
    return shell_tools.run_shell_tool


def ensure_default_tools_registered() -> None:
    """Trigger lazy imports so default tools self-register with the global registry.

    Safe to call multiple times. This is a lightweight way to ensure tools exist
    in the shared tool_registry before asking for them elsewhere.
    """
    try:
        _ = get_read_file_tool()  # noqa: F841
        _ = get_write_file_tool()  # noqa: F841
        _ = get_list_directory_tool()  # noqa: F841
    except Exception:
        pass
    try:
        _ = get_text_analysis_tool_instance()  # noqa: F841
        _ = get_code_analysis_tool_instance()  # noqa: F841
    except Exception:
        pass
    try:
        _ = get_web_search_tool_instance()  # noqa: F841
        _ = get_http_request_tool_instance()  # noqa: F841
    except Exception:
        pass
    try:
        _ = get_mcp_call_tool()  # noqa: F841
    except Exception:
        pass
    try:
        _ = get_run_shell_tool()  # noqa: F841
    except Exception:
        pass

def __getattr__(name):  # PEP 562 module-level getattr for lazy attributes
    accessor_map = {
        "BaseTool": get_base_tool,
        "tool_registry": get_tool_registry_instance,
        "ensure_default_tools_registered": ensure_default_tools_registered,
        "FileOperationTool": get_file_operation_tool,
        "read_file_tool": get_read_file_tool,
        "write_file_tool": get_write_file_tool,
        "list_directory_tool": get_list_directory_tool,
        "TextAnalysisTool": get_text_analysis_tool,
        "CodeAnalysisTool": get_code_analysis_tool,
        "text_analysis_tool": get_text_analysis_tool_instance,
        "code_analysis_tool": get_code_analysis_tool_instance,
        "WebSearchTool": get_web_search_tool,
        "HttpRequestTool": get_http_request_tool,
        "web_search_tool": get_web_search_tool_instance,
        "http_request_tool": get_http_request_tool_instance,
        "MCPCallTool": get_mcp_call_tool,
        "ShellCommandTool": get_shell_command_tool,
        "run_shell_tool": get_run_shell_tool,
    }
    if name in accessor_map:
        return accessor_map[name]()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "BaseTool",
    "tool_registry",
    "FileOperationTool",
    "read_file_tool",
    "write_file_tool",
    "list_directory_tool",
    "TextAnalysisTool",
    "CodeAnalysisTool",
    "text_analysis_tool",
    "code_analysis_tool",
    "WebSearchTool",
    "HttpRequestTool",
    "web_search_tool",
    "http_request_tool",
    "MCPCallTool",
    "ShellCommandTool",
    "run_shell_tool",
]
