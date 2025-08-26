import pytest

from agentsmcp.config import Config, MCPServerConfig
from agentsmcp.tools.mcp_tool import MCPCallTool
from agentsmcp.mcp.manager import MCPManager


@pytest.mark.asyncio
async def test_mcpcalltool_aexecute_awaits_manager(monkeypatch):
    cfg = Config()
    # Minimal MCP server so the manager won't reject by name
    cfg.mcp = [MCPServerConfig(name="dummy", enabled=True, transport="stdio", command=["echo", "hi"])]

    async def fake_call_tool(self, server_name: str, tool_name: str, args: dict):  # type: ignore[no-redef]
        assert server_name == "dummy"
        assert tool_name == "tool"
        assert args == {"x": 1}
        return "ok"

    monkeypatch.setattr(MCPManager, "call_tool", fake_call_tool, raising=False)

    tool = MCPCallTool(cfg)
    result = await tool.aexecute("dummy", "tool", {"x": 1})
    assert result == "ok"

