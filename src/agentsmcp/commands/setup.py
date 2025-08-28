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

PROVIDERS = ["openai", "anthropic", "ollama"]
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "ollama": "mistral",
}

# These are only used for the UI – the wizard will try to auto‑discover
# the real list for Ollama; the others are static references.
MODEL_DEFINITIONS = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4o-20240725",
        "gpt-3.5-turbo-0125",
    ],
    "anthropic": [
        "claude-3.5-sonnet-20240620",
        "claude-3-haiku-20240307",
    ],
    "ollama": ["mistral", "mixtral", "phi3", "llama2"],
}

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
    provider = cfg_data.get("provider") if not reset else None
    detected_keys = _detect_existing_keys()

    if not provider:
        # auto‑detect from env vars and installed services
        if "openai" in detected_keys:
            provider = "openai"
            click.echo("🔍 [auto‑detect] Found OPENAI_API_KEY in environment")
        elif "anthropic" in detected_keys:
            provider = "anthropic"
            click.echo("🔍 [auto‑detect] Found ANTHROPIC_API_KEY in environment")
        elif _is_ollama_installed():
            provider = "ollama"
            click.echo("🔍 [auto‑detect] Local Ollama installation detected")

    # if we still don't know – ask the user
    if not provider:
        click.echo("\n🤖 Choose your AI provider:")
        for i, p in enumerate(PROVIDERS, 1):
            status = ""
            if p == "openai" and "openai" in detected_keys:
                status = " ✅ (API key detected)"
            elif p == "anthropic" and "anthropic" in detected_keys:
                status = " ✅ (API key detected)"
            elif p == "ollama" and _is_ollama_installed():
                status = " ✅ (Ollama installed)"
            click.echo(f"   {i}. {p.title()}{status}")
        
        provider = click.prompt(
            "\nSelect provider",
            type=click.Choice(PROVIDERS, case_sensitive=False),
            default="openai",
            show_choices=False,
        )
    
    cfg_data["provider"] = provider
    click.echo(f"✅ Selected provider: {provider}\n")

    # ---- 3️⃣  API key (if required) -------------------------
    api_key = cfg_data.get("api_key") if not reset else None
    
    if provider in ("openai", "anthropic"):
        env_key = detected_keys.get(provider)
        
        if env_key and not api_key:
            click.echo(f"🔑 Using {provider.upper()}_API_KEY from environment")
            api_key = env_key
        elif not api_key:
            click.echo(f"\n🔐 {provider.title()} API key required")
            click.echo(f"   Get your key from: https://{'platform.openai.com/api-keys' if provider == 'openai' else 'console.anthropic.com'}")
            
            api_key = click.prompt(
                f"Enter {provider.title()} API key",
                hide_input=True,
                confirmation_prompt=True,
            )
        
        cfg_data["api_key"] = api_key

    # ---- 4️⃣  Model selection --------------------------------
    # Start with the static list for the chosen provider
    available_models = MODEL_DEFINITIONS.get(provider, [])

    # Replace Ollama list with a live query if possible
    if provider == "ollama":
        click.echo("🔍 Checking for locally installed Ollama models...")
        live_models = _list_ollama_models()
        if live_models:
            available_models = live_models
            click.echo(f"   Found {len(live_models)} local models")
        else:
            click.echo("   Using default model list (Ollama may not be running)")

    if not available_models:
        # In the very unlikely case that both static and live lists failed
        click.echo(f"⚠️  No {provider} models are known. Using default: {DEFAULT_MODELS[provider]}")
        available_models = [DEFAULT_MODELS[provider]]

    # Show the user a choice – use the default model if we already have one
    model = cfg_data.get("model") if not reset else None
    if model not in available_models:
        model = None  # force a prompt

    if not model:
        click.echo(f"\n🧠 Available {provider} models:")
        for i, m in enumerate(available_models[:5], 1):  # Show top 5 models
            default_marker = " (recommended)" if m == DEFAULT_MODELS[provider] else ""
            click.echo(f"   {i}. {m}{default_marker}")
        
        if len(available_models) > 5:
            click.echo(f"   ... and {len(available_models) - 5} more")
        
        model = click.prompt(
            f"Choose a model",
            type=click.Choice(available_models),
            default=DEFAULT_MODELS[provider],
            show_choices=False,
        )
    
    cfg_data["model"] = model
    click.echo(f"✅ Selected model: {model}\n")

    # ---- 5️⃣ Preference profile -----------------------------
    profile = cfg_data.get("profile") if not reset else None

    if not profile:
        click.echo("⚙️  Choose your preference profile:")
        for name, details in PREFERENCES.items():
            click.echo(f"   • {name}: {details['description']}")
        
        profile = click.prompt(
            "\nSelect profile",
            type=click.Choice(list(PREFERENCES.keys())),
            default="Balanced",
            show_choices=False,
        )
    
    cfg_data["profile"] = profile
    # Apply profile settings
    cfg_data.update(PREFERENCES[profile])
    del cfg_data["description"]  # Remove description from saved config
    click.echo(f"✅ Profile: {profile}\n")

    # ---- 6️⃣ RAG configuration (optional) ------------------
    rag_enabled = cfg_data.get("rag", {}).get("enabled", False) if not reset else False
    
    if not rag_enabled:
        click.echo("🧠 Retrieval Augmented Generation (RAG):")
        click.echo("   RAG can enhance responses with your own documents/knowledge,")
        click.echo("   but requires maintenance to keep knowledge up-to-date.")
        click.echo("   ⚠️  Warning: RAG is disabled by default to avoid stale data issues")
        
        enable_rag = click.confirm("\n   Enable RAG? (recommended: No for beginners)", default=False)
        
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