import yaml
from fastapi.testclient import TestClient

from agentsmcp.server import create_app
from agentsmcp.config import Config


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data))


def test_mcp_endpoints_not_registered_when_disabled(tmp_path, monkeypatch):
    # Prevent writing to user home by redirecting default config path
    monkeypatch.setattr(Config, "default_config_path", lambda: tmp_path / "default.yaml")

    cfg_path = tmp_path / "disabled.yaml"
    write_yaml(cfg_path, {"server": {"host": "127.0.0.1", "port": 8001}, "mcp_api_enabled": False})

    app = create_app(config_path=str(cfg_path))
    client = TestClient(app)

    resp = client.get("/mcp")
    assert resp.status_code == 404


def test_mcp_endpoints_registered_and_mutable_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "default_config_path", lambda: tmp_path / "default.yaml")

    cfg_path = tmp_path / "enabled.yaml"
    write_yaml(cfg_path, {"server": {"host": "127.0.0.1", "port": 8002}, "mcp_api_enabled": True, "mcp": []})

    app = create_app(config_path=str(cfg_path))
    client = TestClient(app)

    # Initially empty; ensure flags object present
    r = client.get("/mcp")
    assert r.status_code == 200
    data = r.json()
    assert data.get("servers") == []
    assert "flags" in data

    # Add server
    payload = {
        "action": "add",
        "name": "git-mcp",
        "transport": "stdio",
        "command": ["npx", "-y", "@modelcontextprotocol/server-git"],
        "enabled": True,
    }
    r2 = client.put("/mcp", json=payload)
    assert r2.status_code == 200
    assert r2.json().get("ok") is True

    # Confirm presence
    r3 = client.get("/mcp")
    assert r3.status_code == 200
    data = r3.json()
    assert any(s.get("name") == "git-mcp" for s in data.get("servers", []))
