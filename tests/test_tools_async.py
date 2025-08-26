import pytest

from agentsmcp.tools.base_tools import BaseTool, tool_registry
from agentsmcp.conversation.llm_client import LLMClient


class DummyAsyncTool(BaseTool):
    def __init__(self):
        super().__init__(name="dummy_async", description="")

    async def aexecute(self, **kwargs):  # type: ignore[override]
        return "async result"

    def execute(self, **kwargs):  # pragma: no cover
        raise RuntimeError("sync execute should never be called for DummyAsyncTool")

    def get_parameters_schema(self):  # type: ignore[override]
        return {"type": "object", "properties": {}}


class DummySyncTool(BaseTool):
    def __init__(self):
        super().__init__(name="dummy_sync", description="")

    def execute(self, **kwargs):  # type: ignore[override]
        return "sync result"

    def get_parameters_schema(self):  # type: ignore[override]
        return {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_execute_tool_prefers_async_and_falls_back_to_thread():
    # Register temporary tools
    tool_registry.register(DummyAsyncTool())
    tool_registry.register(DummySyncTool())
    try:
        client = LLMClient()

        # Async tool – should hit aexecute directly
        async_res = await client._execute_tool("dummy_async", {})
        assert async_res == "async result"

        # Sync tool – should be executed via asyncio.to_thread
        sync_res = await client._execute_tool("dummy_sync", {})
        assert sync_res == "sync result"
    finally:
        # Cleanup
        from agentsmcp.tools.base_tools import tool_registry as _reg

        _reg._tools.pop("dummy_async", None)
        _reg._tools.pop("dummy_sync", None)
