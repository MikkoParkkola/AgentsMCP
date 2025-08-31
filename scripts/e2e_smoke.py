"""Minimal E2E smoke (P2).

Starts a FastAPI app instance and checks a couple of endpoints.
This avoids network binds and runs against in-memory server.
"""

from agentsmcp.server import AgentServer
from agentsmcp.config import Config, AgentConfig
from fastapi.testclient import TestClient


def main() -> int:
    cfg = Config()
    cfg.agents["test"] = AgentConfig(type="test", model="dummy")
    server = AgentServer(cfg)
    client = TestClient(server.app)
    r = client.get("/health")
    assert r.status_code == 200
    r = client.get("/agents")
    assert r.status_code == 200
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

