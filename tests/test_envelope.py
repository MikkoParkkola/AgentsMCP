import pytest
from fastapi.testclient import TestClient

from agentsmcp.server import create_app
from agentsmcp.models import EnvelopeParser, EnvelopeStatus


def client():
    app = create_app()
    return TestClient(app)


def test_envelope_parser_raw_payload():
    payload = {"a": 1}
    data, meta = EnvelopeParser.parse_body(payload)
    assert data == payload
    assert meta.request_id
    assert meta.version == "1.0"


def test_envelope_parser_enveloped_payload():
    env = EnvelopeParser.build_envelope({"b": 2})
    body = env.model_dump(mode="json")
    data, meta = EnvelopeParser.parse_body(body)
    assert data == {"b": 2}
    assert meta.request_id == body["meta"]["request_id"]


def test_build_envelope_success():
    env = EnvelopeParser.build_envelope({"ok": True})
    d = env.model_dump(mode="json")
    assert d["status"] == EnvelopeStatus.SUCCESS
    assert d["payload"] == {"ok": True}
    assert "timestamp" in d["meta"]


def test_health_is_enveloped_by_default():
    c = client()
    r = c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert set(["status", "meta", "payload", "errors"]).issubset(data.keys())
    assert data["status"] == "success"
    assert data["payload"]["status"] == "healthy"


def test_coord_ping_legacy_mode():
    c = client()
    r = c.get("/coord/ping?legacy=true")
    assert r.status_code == 200
    assert r.json() == {"pong": True}


def test_status_not_found_is_error_envelope():
    c = client()
    r = c.get("/status/does-not-exist")
    assert r.status_code == 404
    data = r.json()
    assert data["status"] == "error"
    assert isinstance(data.get("errors"), list)
    assert len(data["errors"]) >= 1
