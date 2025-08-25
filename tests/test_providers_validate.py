import types

from agentsmcp.providers_validate import validate_provider_config
from agentsmcp.config import ProviderType, ProviderConfig


class DummyResp:
    def __init__(self, code=200):
        self.status_code = code


def test_validate_openai_missing_key(monkeypatch):
    cfg = ProviderConfig(name=ProviderType.OPENAI)
    res = validate_provider_config(ProviderType.OPENAI, cfg)
    assert not res.ok and res.reason == "missing_api_key"


def test_validate_ollama_ok(monkeypatch):
    # Monkeypatch httpx.Client.get to simulate a 200 response without network
    import agentsmcp.providers_validate as pv

    def fake_get(self, url, headers=None):  # noqa: ARG002
        return DummyResp(200)

    class FakeClient:
        def __init__(self, timeout=None):  # noqa: ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        get = fake_get

    monkeypatch.setattr(pv, "httpx", types.SimpleNamespace(Client=FakeClient))

    cfg = ProviderConfig(name=ProviderType.OLLAMA, api_base="http://localhost:11434")
    res = validate_provider_config(ProviderType.OLLAMA, cfg)
    assert res.ok

