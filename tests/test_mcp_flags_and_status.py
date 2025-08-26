import yaml
from fastapi.testclient import TestClient

from agentsmcp.server import create_app
from agentsmcp.config import Config


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data))


def test_mcp_flags_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "default_config_path", lambda: tmp_path / "default.yaml")

    cfg_path = tmp_path / "config.yaml"
    write_yaml(
        cfg_path,
        {
            "mcp_api_enabled": True,
            "mcp_stdio_enabled": True,
            "mcp_ws_enabled": False,
            "mcp_sse_enabled": False,
        },
    )

    app = create_app(config_path=str(cfg_path))
    client = TestClient(app)

    # GET /mcp returns flags
    r = client.get("/mcp")
    assert r.status_code == 200
    data = r.json()
    assert data.get("flags") == {"stdio": True, "ws": False, "sse": False}

    # PUT /mcp set_flags ws=true
    r2 = client.put("/mcp", json={"action": "set_flags", "ws": True})
    assert r2.status_code == 200
    assert r2.json().get("ok") is True

    # flags now reflect ws True
    r3 = client.get("/mcp")
    assert r3.status_code == 200
    assert r3.json().get("flags", {}).get("ws") is True

    # /mcp/status includes manager and servers
    r4 = client.get("/mcp/status")
    assert r4.status_code == 200
    st = r4.json()
    assert "manager" in st and "servers" in st
    mgr = st["manager"]
    # All expected keys present
    assert set(mgr.keys()) == {"ttl_seconds", "max_retries", "base_delay"}
