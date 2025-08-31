import importlib
import sys


def test_tools_lazy_module_does_not_import_submodules_on_import():
    # Clean slate
    for k in list(sys.modules.keys()):
        if k.startswith('agentsmcp.tools'):
            del sys.modules[k]

    import agentsmcp.tools as tools

    # Submodules should not be imported yet
    assert 'agentsmcp.tools.file_tools' not in sys.modules
    assert 'agentsmcp.tools.web_tools' not in sys.modules

    # Trigger registration
    tools.ensure_default_tools_registered()

    # Now submodules should be imported when accessed
    assert tools.tool_registry is not None


def test_ui_lazy_imports():
    for k in list(sys.modules.keys()):
        if k.startswith('agentsmcp.ui'):
            del sys.modules[k]

    import agentsmcp.ui as ui

    # Heavy submodules not yet loaded
    assert 'agentsmcp.ui.command_interface' not in sys.modules

    # Accessing attribute should import lazily
    _ = ui.CommandInterface  # noqa: F841
    assert 'agentsmcp.ui.command_interface' in sys.modules


def test_providers_list_models_caches(monkeypatch):
    from agentsmcp.providers import list_models, ProviderType, ProviderConfig, _MODELS_CACHE

    calls = {"count": 0}

    def fake_list(config):
        calls["count"] += 1
        return []

    # Patch one provider function
    import agentsmcp.providers as providers
    monkeypatch.setattr(providers, 'openai_list_models', fake_list)

    cfg = ProviderConfig(name=ProviderType.OPENAI, api_key='k', api_base='https://api.openai.com/v1')
    _MODELS_CACHE.clear()
    list_models(ProviderType.OPENAI, cfg)
    list_models(ProviderType.OPENAI, cfg)
    assert calls["count"] == 1  # cached second call

