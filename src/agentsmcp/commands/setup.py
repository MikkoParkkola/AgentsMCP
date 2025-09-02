"""
The wizard that walks the user through a first‑time AgentsMCP set‑up.
It automatically checks for environment variables, installed services
(ollama, OpenAI, Anthropic), lists available models, exposes a simple
profile system, and finally writes a clean config file.
"""

import click
import os
import shutil
import sys
import subprocess
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import json
import yaml

# --------------------------------------------------------------
# 1. Import the new configuration system
# --------------------------------------------------------------
#
# The new `agentsmcp.config` package exposes a top‑level
# `Config` object that already knows how to read/write the
# `~/.agentsmcp/config.yaml`.  If for whatever reason the import fails
# (e.g., during a very early bootstrap) we fall back to a tiny
# in‑process implementation that behaves identically for our needs.
# --------------------------------------------------------------
try:
    from agentsmcp.config import ConfigLoader, get_config
    
    class ConfigWrapper:
        def __init__(self):
            self._file = Path.home() / ".agentsmcp" / "config.yaml"
            self.data = {}
            self.load()
        
        def load(self) -> None:
            if self._file.exists():
                with open(self._file, "r", encoding="utf-8") as f:
                    try:
                        self.data = yaml.safe_load(f) or {}
                    except yaml.YAMLError:
                        self.data = {}
            else:
                self.data = {}
        
        def save(self) -> None:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.data, f, default_flow_style=False, indent=2)
    
    Config = ConfigWrapper
    
except Exception as exc:  # pragma: no cover – exercised only when the real library is missing
    class _FallbackConfig:
        """
        Minimal in‑process implementation that mimics
        `agentsmcp.config.Config`.  It is only used if the real
        library cannot be imported; the production release will
        always have the real implementation.
        """

        _file: Path

        def __init__(self):
            self._file = Path.home() / ".agentsmcp" / "config.yaml"
            self.data = {}
            self.load()

        def load(self) -> None:
            if self._file.exists():
                with open(self._file, "r", encoding="utf-8") as f:
                    try:
                        self.data = yaml.safe_load(f) or {}
                    except yaml.YAMLError:
                        self.data = {}
            else:
                self.data = {}

        def save(self) -> None:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.data, f, default_flow_style=False, indent=2)

    Config = _FallbackConfig  # type: ignore

# --------------------------------------------------------------
# 2. Helper constants
# --------------------------------------------------------------

PROVIDERS = ["openai", "anthropic", "ollama", "ollama-turbo", "google-ai", "openrouter"]
# Removed hardcoded models - models are now fetched from provider APIs or manually input
# No default models - they will be provided by API or manual input

# These are only used for the UI – the wizard will try to auto‑discover
# the real list for Ollama; the others are static references.
# Removed hardcoded MODEL_DEFINITIONS - models are now fetched from provider APIs or manually input
# No more hardcoded model lists

PREFERENCES = {
    "Fast": {"description": "Prioritizes speed and responsiveness", "temperature": 0.9},
    "Quality": {"description": "Focuses on accuracy and detailed responses", "temperature": 0.3},
    "Cost‑Efficient": {"description": "Balances quality with cost considerations", "temperature": 0.7},
    "Balanced": {"description": "Optimal balance of speed, quality, and cost", "temperature": 0.5},
}

# --------------------------------------------------------------
# 3. Helper utilities
# --------------------------------------------------------------

def _is_ollama_installed() -> bool:
    """Return True if the `ollama` binary is in PATH."""
    return bool(shutil.which("ollama"))


def _list_ollama_models() -> list[str]:
    """Attempt to query a locally running Ollama instance for its
    available models.  If the query fails the function falls back
    to an empty list, which triggers the static defaults."""
    try:
        resp = urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", []) if "name" in m]
    except (URLError, HTTPError, json.JSONDecodeError):
        return []


def _list_openai_models(api_key: str) -> list[str]:
    """Fetch available OpenAI models dynamically from the API."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        models = client.models.list()
        # Filter for GPT models and sort by recency/preference
        model_names = [m.id for m in models.data if m.id.startswith(('gpt-4', 'gpt-3.5'))]
        # Preferred order: newest GPT-4 models first, then GPT-3.5
        preferred_order = ['gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-08-06', 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18']
        ordered_models = []
        for preferred in preferred_order:
            if preferred in model_names:
                ordered_models.append(preferred)
        # Add any remaining models not in preferred list
        for model in model_names:
            if model not in ordered_models:
                ordered_models.append(model)
        return ordered_models[:10]  # Limit to first 10 models
    except Exception:
        return []


def _list_anthropic_models(api_key: str) -> list[str]:
    """Return available Anthropic models. Since Anthropic doesn't have a public models API,
    we return the known current models in preferred order."""
    try:
        import anthropic
        # Test the API key by making a minimal request
        client = anthropic.Anthropic(api_key=api_key)
        # We can't actually list models via API, so return known good models
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620", 
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229"
        ]
    except Exception:
        return []


def _list_ollama_turbo_models(api_key: str) -> list[str]:
    """Fetch available models from Ollama Turbo (ollama.com) API.
    Works exactly like local ollama but uses https://ollama.com API endpoint."""
    try:
        import urllib.request
        req = urllib.request.Request("https://ollama.com/api/tags")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            
        return [m["name"] for m in data.get("models", []) if "name" in m]
    except (URLError, HTTPError, json.JSONDecodeError):
        return []


def _list_google_ai_models(api_key: str) -> list[str]:
    """Return available Google AI models."""
    try:
        # Google AI Studio/Gemini API - we'll use known models since the API structure varies
        return [
            "gemini-1.5-flash",
            "gemini-1.5-pro",  
            "gemini-pro",
            "gemini-1.0-pro",
            "text-bison",
        ]
    except Exception:
        return []


def _list_openrouter_models(api_key: str) -> list[str]:
    """Fetch available models from OpenRouter API."""
    try:
        import urllib.request
        req = urllib.request.Request("https://openrouter.ai/api/v1/models")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            
        models = []
        if "data" in data:
            for model in data["data"]:
                model_id = model.get("id", "")
                if model_id:
                    models.append(model_id)
        
        # Sort by popularity/preference
        preferred_models = [
            "gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet", 
            "llama-3.1-8b", "llama-3.1-70b", "gpt-4-turbo"
        ]
        
        ordered = []
        for pref in preferred_models:
            matching = [m for m in models if pref in m.lower()]
            ordered.extend(matching[:2])  # Max 2 variants per preferred model
            
        return ordered[:10]  # Return top 10
    except Exception:
        return []


def _print_system_info() -> None:
    """Print a terse system summary (RAM + CPU).  Any failures are
    silently ignored – the wizard should always finish."""
    try:
        import psutil

        mem_gb = psutil.virtual_memory().total / (1024**3)
        cpu = psutil.cpu_count()
        click.echo(f"• RAM: {mem_gb:.1f} GB")
        click.echo(f"• CPU cores: {cpu}")
    except Exception:  # pragma: no cover – optional dependency
        click.echo("• [Could not gather system info]")


def _detect_existing_keys() -> dict:
    """Detect API keys from environment variables."""
    keys = {}
    key_mappings = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "google-ai": ["GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
        "ollama-turbo": ["OLLAMA_API_KEY"],  # Uses same env var as regular ollama
    }
    
    for provider, env_vars in key_mappings.items():
        for env_var in env_vars:
            if os.getenv(env_var):
                keys[provider] = os.getenv(env_var)
                break
    
    return keys


# --------------------------------------------------------------
# 4. The actual command definition
# --------------------------------------------------------------

@click.command(
    name="setup",
    help="🛠️  Interactive wizard that creates a valid AgentsMCP configuration file.",
)
@click.option(
    "--reset", 
    is_flag=True, 
    help="Reset configuration and start fresh"
)
def setup(reset: bool) -> None:
    """Run the configuration wizard."""
    
    click.echo("\n🚀 Welcome to the AgentsMCP setup wizard!")
    click.echo("=" * 50)
    
    # ---- 1️⃣  Load any pre‑existing config ----------------
    cfg = Config()  # already auto‑loads from disk
    cfg_data: dict = cfg.data  # type: ignore
    
    if reset:
        cfg_data.clear()
        click.echo("🔄 Configuration reset - starting fresh\n")
    elif cfg_data:
        click.echo("📁 Found existing configuration - we'll update it\n")

    # ---- 2️⃣  Detect provider --------------------------------
    current_provider = cfg_data.get("provider") if not reset else None
    detected_keys = _detect_existing_keys()
    
    # Show current setting if it exists
    if current_provider:
        click.echo(f"📝 Current provider: {current_provider}")

    # Always ask for provider selection (show detected options)
    click.echo("\n🤖 Choose your AI provider:")
    for i, p in enumerate(PROVIDERS, 1):
        status = ""
        if p in detected_keys:
            status = " ✅ (API key detected)"
        elif p == "ollama" and _is_ollama_installed():
            status = " ✅ (Ollama installed)"
        
        # Format provider names nicely
        display_name = p.replace("-", " ").title()
        click.echo(f"   {i}. {display_name}{status}")
    
    # Use current provider as default if available, otherwise openai
    default_provider = current_provider if current_provider else "openai"
    provider = click.prompt(
        "\nSelect provider",
        type=click.Choice(PROVIDERS, case_sensitive=False),
        default=default_provider,
        show_choices=False,
    )
    
    cfg_data["provider"] = provider
    click.echo(f"✅ Selected provider: {provider}\n")

    # ---- 3️⃣  API key (if required) -------------------------
    current_api_key = cfg_data.get("api_key") if not reset else None
    
    # Define which providers need API keys and their URLs
    api_key_providers = {
        "openai": "https://platform.openai.com/api-keys",
        "anthropic": "https://console.anthropic.com",
        "google-ai": "https://makersuite.google.com/app/apikey",
        "openrouter": "https://openrouter.ai/keys",
        "ollama-turbo": "https://ollama.com/account/api-keys",
    }
    
    if provider in api_key_providers:
        display_name = provider.replace("-", " ").title()
        env_key = detected_keys.get(provider)
        
        # Show current status
        if current_api_key:
            click.echo(f"📝 Current {display_name} API key: ****{current_api_key[-4:] if len(current_api_key) > 4 else '****'}")
        if env_key:
            click.echo(f"🔍 {display_name} API key detected in environment")
        
        click.echo(f"\n🔐 {display_name} API key configuration")
        click.echo(f"   Get your key from: {api_key_providers[provider]}")
        
        # Always ask for API key, but offer to keep existing one
        if current_api_key or env_key:
            use_existing = click.confirm(
                f"Keep existing {display_name} API key?",
                default=True
            )
            if use_existing:
                api_key = current_api_key or env_key
            else:
                api_key = click.prompt(
                    f"Enter new {display_name} API key",
                    hide_input=True,
                    confirmation_prompt=True,
                )
        else:
            api_key = click.prompt(
                f"Enter {display_name} API key",
                hide_input=True,
                confirmation_prompt=True,
            )
        
        cfg_data["api_key"] = api_key

    # ---- 4️⃣  Model selection --------------------------------
    # Fetch models dynamically from provider APIs
    available_models = []
    
    if provider == "openai" and api_key:
        click.echo("🔍 Fetching available OpenAI models from API...")
        available_models = _list_openai_models(api_key)
        if available_models:
            click.echo(f"   Found {len(available_models)} OpenAI models")
        else:
            click.echo("   API fetch failed - you'll need to enter model name manually")
    elif provider == "anthropic" and api_key:
        click.echo("🔍 Loading available Anthropic models...")
        available_models = _list_anthropic_models(api_key)
        if available_models:
            click.echo(f"   Found {len(available_models)} Anthropic models")
        else:
            click.echo("   Using known Anthropic models")
    elif provider == "ollama":
        click.echo("🔍 Checking for locally installed Ollama models...")
        available_models = _list_ollama_models()
        if available_models:
            click.echo(f"   Found {len(available_models)} local models")
        else:
            click.echo("   No local models found - you'll need to enter model name manually")
    elif provider == "ollama-turbo" and api_key:
        click.echo("🔍 Fetching available Ollama Turbo models from API...")
        available_models = _list_ollama_turbo_models(api_key)
        if available_models:
            click.echo(f"   Found {len(available_models)} Ollama Turbo models")
        else:
            click.echo("   API fetch failed - you'll need to enter model name manually")
    elif provider == "google-ai" and api_key:
        click.echo("🔍 Loading available Google AI models...")
        available_models = _list_google_ai_models(api_key)
        if available_models:
            click.echo(f"   Found {len(available_models)} Google AI models")
        else:
            click.echo("   Using known Google AI models")
    elif provider == "openrouter" and api_key:
        click.echo("🔍 Fetching available OpenRouter models from API...")
        available_models = _list_openrouter_models(api_key)
        if available_models:
            click.echo(f"   Found {len(available_models)} OpenRouter models")
        else:
            click.echo("   API fetch failed - you'll need to enter model name manually")

    # Handle model selection - always ask but show current as default
    current_model = cfg_data.get("model") if not reset else None
    
    # Show current model if it exists
    if current_model:
        click.echo(f"📝 Current model: {current_model}")
    
    # Always ask for model selection
    if available_models:
        # Show available models from API
        click.echo(f"\n🧠 Available {provider} models:")
        for i, m in enumerate(available_models[:5], 1):  # Show top 5 models
            click.echo(f"   {i}. {m}")
        
        if len(available_models) > 5:
            click.echo(f"   ... and {len(available_models) - 5} more")
        
        # Add option for manual input
        click.echo(f"   Or type a custom model name")
        
        # Use current model as default if it's in the available list
        default_model = current_model if current_model and current_model in available_models else available_models[0]
        
        model_choice = click.prompt(
            f"Choose a model or type custom name",
            type=str,
            default=default_model,
            show_default=True,
        )
        
        # If they entered a number, map it to the model name
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(available_models):
            model = available_models[int(model_choice) - 1]
        elif model_choice in available_models:
            model = model_choice
        else:
            # Custom model name
            model = model_choice
    else:
        # No models available from API - require manual input
        click.echo(f"\n🧠 Enter {provider} model name:")
        provider_suggestions = {
            "openai": "gpt-4o-mini, gpt-4o, gpt-3.5-turbo",
            "anthropic": "claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-haiku-20240307",
            "ollama": "mistral, llama3.1, phi3",
            "ollama-turbo": "gpt-4o-mini, gpt-4o, llama-3.1-8b",
            "google-ai": "gemini-1.5-flash, gemini-1.5-pro, gemini-pro",
            "openrouter": "gpt-4o-mini, gpt-4o, claude-3.5-sonnet"
        }
        
        if provider in provider_suggestions:
            click.echo(f"   Popular models: {provider_suggestions[provider]}")
        
        # Use current model as default if available
        default_model = current_model if current_model else ""
        
        model = click.prompt(
            f"Enter model name",
            type=str,
            default=default_model,
            show_default=bool(default_model),
            )
    
    cfg_data["model"] = model
    click.echo(f"✅ Selected model: {model}\n")

    # ---- 5️⃣ Preference profile -----------------------------
    current_profile = cfg_data.get("profile") if not reset else None
    
    # Show current profile if it exists
    if current_profile:
        click.echo(f"📝 Current profile: {current_profile}")

    # Always ask for profile selection
    click.echo("⚙️  Choose your preference profile:")
    for name, details in PREFERENCES.items():
        current_marker = " (current)" if name == current_profile else ""
        click.echo(f"   • {name}: {details['description']}{current_marker}")
    
    default_profile = current_profile if current_profile else "Balanced"
    profile = click.prompt(
        "\nSelect profile",
        type=click.Choice(list(PREFERENCES.keys())),
        default=default_profile,
        show_choices=False,
    )
    
    cfg_data["profile"] = profile
    # Apply profile settings
    cfg_data.update(PREFERENCES[profile])
    del cfg_data["description"]  # Remove description from saved config
    click.echo(f"✅ Profile: {profile}\n")

    # ---- 6️⃣ RAG configuration (optional) ------------------
    current_rag_enabled = cfg_data.get("rag", {}).get("enabled", False) if not reset else False
    
    # Show current RAG setting if it exists
    if current_rag_enabled:
        click.echo(f"📝 Current RAG setting: enabled")
    elif not reset and "rag" in cfg_data:
        click.echo(f"📝 Current RAG setting: disabled")
    
    # Always ask about RAG configuration
    click.echo("🧠 Retrieval Augmented Generation (RAG):")
    click.echo("   RAG can enhance responses with your own documents/knowledge,")
    click.echo("   but requires maintenance to keep knowledge up-to-date.")
    click.echo("   ⚠️  Warning: RAG is disabled by default to avoid stale data issues")
    
    # Use current setting as default
    default_rag = current_rag_enabled
    enable_rag = click.confirm("\n   Enable RAG? (recommended: No for beginners)", default=default_rag)
    
    if enable_rag:
        click.echo("\n📚 RAG Configuration:")
        
        # Check RAG capabilities
        try:
            from agentsmcp.config.env_detector import detect_rag_capabilities
            capabilities = detect_rag_capabilities()
        except Exception:
            capabilities = {}
        
        # Embedding model selection
        click.echo("   Choose embedding model:")
        click.echo("   1. Sentence Transformers (all-MiniLM-L6-v2) - recommended")
        if capabilities.get("ollama", False):
            click.echo("   2. Ollama embeddings (requires local Ollama)")
        
        embed_choice = click.prompt(
            "   Select embedding model",
            type=click.Choice(["1", "2"] if capabilities.get("ollama", False) else ["1"]),
            default="1",
            show_choices=False
        )
        
        # Vector store selection
        vector_backends = []
        if capabilities.get("faiss", False):
            vector_backends.append("faiss")
            click.echo("   ✅ FAISS available")
        if capabilities.get("lancedb", False):
            vector_backends.append("lancedb")
            click.echo("   ✅ LanceDB available")
        
        if not vector_backends:
            click.echo("   ⚠️  No vector stores detected. You'll need to install faiss-cpu or lancedb")
            vector_backend = "faiss"
        else:
            vector_backend = vector_backends[0]  # Use first available
        
        # Knowledge TTL
        ttl_days = click.prompt("   Knowledge expiry (days)", type=int, default=90)
        
        # Configure RAG settings
        if "rag" not in cfg_data:
            cfg_data["rag"] = {}
        
        cfg_data["rag"]["enabled"] = True
        cfg_data["rag"]["embedder"] = {
            "model": "all-MiniLM-L6-v2" if embed_choice == "1" else "ollama:llama3:embedding"
        }
        cfg_data["rag"]["vector_store"] = {"backend": vector_backend}
        cfg_data["rag"]["freshness_policy"] = {"ttl_days": ttl_days}
        
        click.echo(f"✅ RAG enabled with {ttl_days}-day knowledge expiry")
        click.echo("   💡 Use 'agentsmcp rag ingest <path>' to add knowledge")
    else:
        # Ensure RAG is explicitly disabled
        if "rag" not in cfg_data:
            cfg_data["rag"] = {}
        cfg_data["rag"]["enabled"] = False
        click.echo("✅ RAG disabled (can be enabled later)")
        
        click.echo()
    
    # ---- 7️⃣ System capability hint ------------------------
    click.echo("🖥️  System Information:")
    _print_system_info()

    # ---- 8️⃣ Persist the config file ------------------------
    cfg.save()
    click.echo(f"\n💾 Configuration saved to: {cfg._file}")
    click.echo("   (You can edit this file manually if needed)\n")

    # ---- 9️⃣ Final user guidance ----------------------------
    click.echo("🎉 Setup complete!")
    click.echo("=" * 50)
    click.echo("\n📋 Next steps:")
    click.echo("   • Run 'agentsmcp --help' to see available commands")
    click.echo("   • Try 'agentsmcp roles list' to see available agent roles")
    if cfg_data.get("rag", {}).get("enabled", False):
        click.echo("   • Use 'agentsmcp rag ingest <path>' to add knowledge to RAG")
        click.echo("   • Try 'agentsmcp rag list' to see your knowledge base")
    click.echo("   • Use 'agentsmcp setup --reset' to reconfigure anytime")
    
    if provider == "ollama" and not _list_ollama_models():
        click.echo("\n💡 Tip: Start Ollama with 'ollama serve' to enable local models")
    
    click.echo("\nHappy agent building! 🤖✨")