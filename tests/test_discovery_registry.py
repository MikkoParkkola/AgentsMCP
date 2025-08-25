from pathlib import Path
import json


def test_registry_write_and_list(monkeypatch, tmp_path):
    # Redirect registry path to a temp location
    from agentsmcp.discovery import registry as reg
    temp_file = tmp_path / "registry.json"
    monkeypatch.setattr(reg, "REGISTRY_PATH", temp_file)

    e = reg.Entry(
        agent_id="a1",
        name="agentsmcp",
        capabilities=["codex"],
        transport="http",
        endpoint="http://localhost:8000",
        token=None,
    )
    reg.write_entry(e)
    assert temp_file.exists()
    items = reg.list_entries()
    assert any(x.agent_id == "a1" for x in items)

