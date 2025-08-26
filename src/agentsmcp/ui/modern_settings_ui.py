"""
AgentsMCP – Modern Settings UI
-----------------------------
A beautiful, themed settings interface that matches the main CLI app's design.
Uses full terminal width and modern UI components for a professional experience.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import shutil

from .theme_manager import ThemeManager
from .ui_components import UIComponents
from .keyboard_input import KeyboardInput, MenuSelector, KeyCode


class ModernSettingsUI:
    """Modern, themed settings interface that matches the CLI app design"""
    
    def __init__(self, theme_manager: Optional[ThemeManager] = None, ui_components: Optional[UIComponents] = None):
        self.theme_manager = theme_manager or ThemeManager()
        self.ui = ui_components or UIComponents(self.theme_manager)
        self.config_path = Path.home() / ".agentsmcp" / "config.json"
        
        # Keyboard input handling
        self.keyboard = KeyboardInput()
        self.menu_selector = MenuSelector(self.keyboard)
        
        # Current settings
        self.current_settings = self._load_current_settings()
        
        # Available providers and models
        self.providers = {
            "ollama-turbo": {
                "name": "Ollama Turbo",
                "models": ["gpt-oss:120b", "gpt-oss:70b", "gpt-oss:20b", "llama3.1:405b", "llama3.1:70b"],
                "default_host": "http://127.0.0.1:11435"
            },
            "ollama": {
                "name": "Ollama",
                "models": ["llama3.1:70b", "llama3.1:8b", "codellama:13b", "gpt-oss:20b"],
                "default_host": "http://localhost:11434"
            },
            "openai": {
                "name": "OpenAI",
                "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "default_host": "https://api.openai.com/v1"
            },
            "anthropic": {
                "name": "Anthropic",
                "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                "default_host": "https://api.anthropic.com/v1"
            }
        }

    def _load_current_settings(self) -> Dict[str, Any]:
        """Load current settings from config file"""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
                # Migration: move legacy top-level api_key into api_keys[provider]
                try:
                    legacy_key = data.get('api_key')
                    if legacy_key:
                        provider = data.get('provider') or 'ollama-turbo'
                        keys = data.get('api_keys') or {}
                        if provider not in keys:
                            keys[provider] = legacy_key
                            data['api_keys'] = keys
                        # remove legacy field and persist
                        data.pop('api_key', None)
                        self._save_settings(data)
                except Exception:
                    pass
                return data
            except (json.JSONDecodeError, OSError):
                pass
        
        # Default settings
        return {
            "provider": "ollama-turbo",
            "model": "gpt-oss:120b", 
            "host": "http://127.0.0.1:11435",
            "temperature": 0.7,
            "max_tokens": 1024,
            # Provider-specific keys map: { provider_name: api_key }
            "api_keys": {}
        }

    def _save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to config file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(settings, indent=2))
            return True
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error saving settings: {e}")
            return False

    def _create_header(self) -> str:
        """Create themed header"""
        header_content = f"""🛠️  AgentsMCP Settings Configuration

Configure your LLM provider, model and generation parameters.
Changes will take effect immediately for new requests."""

        return self.ui.box(
            header_content,
            title="⚙️ Configuration",
            style='heavy',
            width=min(self.ui.terminal_width - 4, 100)
        )

    def _create_provider_selection_display(self, current_provider: str, selected_index: int = 0) -> str:
        """Create provider selection display with arrow key navigation"""
        options_text = []
        providers_list = list(self.providers.items())
        
        for i, (provider_key, provider_info) in enumerate(providers_list):
            # Different indicators for current/selected vs others
            if provider_key == current_provider:
                indicator = "●"  # Current setting
            else:
                indicator = "○"  # Available option
            
            # Highlight selected option
            if i == selected_index:
                # Selected option - use bright highlighting
                line = f"  ▶ {indicator} {provider_info['name']}"
                if provider_key == current_provider:
                    line += f" (host: {self.current_settings.get('host', provider_info['default_host'])})"
                colored_line = self.theme_manager.colorize(line, "primary")
            else:
                # Non-selected options
                color = "text_primary" if provider_key == current_provider else "text_muted"
                line = f"    {indicator} {provider_info['name']}"
                if provider_key == current_provider:
                    line += f" (host: {self.current_settings.get('host', provider_info['default_host'])})"
                colored_line = self.theme_manager.colorize(line, color)
            
            options_text.append(colored_line)
        
        provider_content = f"""Select LLM Provider:

{chr(10).join(options_text)}

Use ↑/↓ arrows to navigate, Enter to select, ESC/q to cancel:"""
        
        return self.ui.box(
            provider_content,
            title="🔌 Provider Selection",
            style='light',
            width=min(self.ui.terminal_width - 4, 90)
        )

    def _create_model_selection_display(self, provider_key: str, current_model: str, selected_index: int = 0) -> str:
        """Create model selection display with arrow key navigation"""
        provider_info = self.providers[provider_key]
        models = provider_info["models"]
        
        options_text = []
        for i, model in enumerate(models):
            # Different indicators for current/selected vs others
            indicator = "●" if model == current_model else "○"
            
            # Highlight selected option
            if i == selected_index:
                # Selected option - use bright highlighting
                line = f"  ▶ {indicator} {model}"
                colored_line = self.theme_manager.colorize(line, "primary")
            else:
                # Non-selected options
                color = "text_primary" if model == current_model else "text_muted"
                line = f"    {indicator} {model}"
                colored_line = self.theme_manager.colorize(line, color)
            
            options_text.append(colored_line)
        
        model_content = f"""Select Model for {provider_info['name']}:

{chr(10).join(options_text)}

Use ↑/↓ arrows to navigate, Enter to select, ESC/q to cancel:"""
        
        return self.ui.box(
            model_content,
            title="🧠 Model Selection",
            style='light',
            width=min(self.ui.terminal_width - 4, 90)
        )

    def _create_parameter_display(self, selected_index: int = 0) -> str:
        """Create parameter configuration display with arrow key navigation"""
        current = self.current_settings
        
        parameters = [
            f"Temperature: {current.get('temperature', 0.7)} (0.0-2.0, creativity control)",
            f"Max Tokens: {current.get('max_tokens', 1024)} (1-4096, response length limit)", 
            f"API Key: {'*' * 8 if (current.get('api_keys', {}).get(current.get('provider',''), '')) else 'Not set'} (for {current.get('provider','provider')})",
            f"Host URL: {current.get('host', 'default')}",
            "Skip parameter configuration"
        ]
        
        options_text = []
        for i, param in enumerate(parameters):
            if i == selected_index:
                # Selected option - use bright highlighting
                line = f"  ▶ {param}"
                colored_line = self.theme_manager.colorize(line, "primary")
            else:
                # Non-selected options
                line = f"    {param}"
                colored_line = self.theme_manager.colorize(line, "text_muted")
            
            options_text.append(colored_line)
        
        param_content = f"""Generation Parameters:

{chr(10).join(options_text)}

Use ↑/↓ arrows to navigate, Enter to select, ESC/q to cancel:"""
        
        return self.ui.box(
            param_content,
            title="⚡ Parameters",
            style='light',
            width=min(self.ui.terminal_width - 4, 90)
        )

    def _create_summary_display(self) -> str:
        """Create settings summary display"""
        current = self.current_settings
        provider_info = self.providers.get(current['provider'], {'name': current['provider']})
        
        has_key = bool(current.get('api_keys', {}).get(current['provider'], ''))
        summary_content = f"""Current Configuration Summary:

🔌 Provider: {provider_info['name']} ({current['provider']})
🧠 Model: {current['model']}
🌐 Host: {current['host']}
⚡ Temperature: {current['temperature']}
📊 Max Tokens: {current['max_tokens']}
🔑 API Key: {'Configured' if has_key else 'Not set'} (for {current['provider']})

Press Enter to save, ESC or 'c' to cancel."""
        
        return self.ui.box(
            summary_content,
            title="✅ Configuration Summary", 
            style='heavy',
            width=min(self.ui.terminal_width - 4, 90)
        )

    def _get_user_input(self, prompt: str) -> str:
        """Get user input with styled prompt (legacy method for text input)"""
        themed_prompt = self.theme_manager.colorize(f"\n{prompt} ", "primary")
        try:
            result = input(themed_prompt).strip()
            return result
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or EOF gracefully
            print("\n")
            return ""
    
    def _wait_for_enter(self) -> None:
        """Wait for user to press Enter with error handling"""
        try:
            input("Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or EOF gracefully
            print("\n")
    
    def _edit_temperature(self) -> bool:
        """Edit temperature parameter"""
        print(self.ui.clear_screen())
        temp_input = self._get_user_input("Enter temperature (0.0-2.0):")
        try:
            temp = float(temp_input)
            if 0.0 <= temp <= 2.0:
                self.current_settings['temperature'] = temp
                return True
            else:
                print(self.theme_manager.colorize("Temperature must be between 0.0 and 2.0", "error"))
                input("Press Enter to continue...")
                return False
        except ValueError:
            print(self.theme_manager.colorize("Please enter a valid number.", "error"))
            input("Press Enter to continue...")
            return False
    
    def _edit_max_tokens(self) -> bool:
        """Edit max tokens parameter"""
        print(self.ui.clear_screen())
        tokens_input = self._get_user_input("Enter max tokens (1-4096):")
        try:
            tokens = int(tokens_input)
            if 1 <= tokens <= 4096:
                self.current_settings['max_tokens'] = tokens
                return True
            else:
                print(self.theme_manager.colorize("Max tokens must be between 1 and 4096", "error"))
                input("Press Enter to continue...")
                return False
        except ValueError:
            print(self.theme_manager.colorize("Please enter a valid number.", "error"))
            input("Press Enter to continue...")
            return False
    
    def _edit_api_key(self) -> bool:
        """Edit API key parameter"""
        print(self.ui.clear_screen())
        key_input = self._get_user_input("Enter API key (leave empty to clear):")
        provider = self.current_settings.get('provider', '')
        keys = self.current_settings.get('api_keys', {})
        if key_input:
            keys[provider] = key_input
        else:
            if provider in keys:
                keys.pop(provider)
        self.current_settings['api_keys'] = keys
        return True
    
    def _edit_host_url(self) -> bool:
        """Edit host URL parameter"""
        print(self.ui.clear_screen())
        current_host = self.current_settings.get('host', '')
        host_input = self._get_user_input(f"Enter host URL [current: {current_host}]:")
        
        # Allow empty input to keep current value
        if host_input.strip():
            self.current_settings['host'] = host_input.strip()
            print(self.theme_manager.colorize(f"✅ Host URL updated to: {host_input.strip()}", 'success'))
            self._wait_for_enter()
            return True
        return True  # Return True even if no change (empty input keeps current)
    
    def _confirm_and_save(self) -> bool:
        """Confirmation dialog with arrow key navigation"""
        while True:
            print(self.ui.clear_screen())
            print(self._create_header())
            print(self._create_summary_display())
            
            # Get keyboard input
            key_code, char = self.keyboard.get_key()
            
            if key_code == KeyCode.ENTER:
                # Save settings
                success = self._save_settings(self.current_settings)
                if success:
                    success_msg = self.ui.box(
                        "✅ Settings saved successfully!\nConfiguration will be used for new requests.",
                        title="Success",
                        style='rounded',
                        width=min(self.ui.terminal_width - 4, 60)
                    )
                    print(success_msg)
                    input("\nPress Enter to continue...")
                    return True
                else:
                    error_msg = self.ui.box(
                        "❌ Failed to save settings.\nPlease check file permissions and try again.",
                        title="Error",
                        style='rounded',
                        width=min(self.ui.terminal_width - 4, 60)
                    )
                    print(error_msg)
                    input("\nPress Enter to continue...")
                    return False
                    
            elif key_code == KeyCode.ESCAPE:
                return False
                
            elif char and char.lower() == 'c':
                return False

    def _select_provider(self) -> bool:
        """Interactive provider selection with arrow key navigation"""
        providers_list = list(self.providers.keys())
        current_provider_index = 0
        
        # Find current provider index
        try:
            current_provider_index = providers_list.index(self.current_settings['provider'])
        except ValueError:
            current_provider_index = 0
        
        selected_index = current_provider_index
        
        while True:
            # Display current state
            print(self.ui.clear_screen())
            print(self._create_header())
            print(self._create_provider_selection_display(self.current_settings['provider'], selected_index))
            
            # Get keyboard input
            key_code, char = self.keyboard.get_key()
            
            if key_code == KeyCode.UP:
                selected_index = (selected_index - 1) % len(providers_list)
                
            elif key_code == KeyCode.DOWN:
                selected_index = (selected_index + 1) % len(providers_list)
                
            elif key_code == KeyCode.ENTER:
                # Select current option
                selected_provider = providers_list[selected_index]
                provider_info = self.providers[selected_provider]
                
                self.current_settings['provider'] = selected_provider
                self.current_settings['host'] = provider_info['default_host']
                # Set first available model as default
                self.current_settings['model'] = provider_info['models'][0]
                return True
                
            elif key_code == KeyCode.ESCAPE:
                return False
                
            elif char and char.lower() == 'q':
                return False

    def _select_model(self) -> bool:
        """Interactive model selection with arrow key navigation"""
        provider_key = self.current_settings['provider']
        provider_info = self.providers[provider_key]
        models = provider_info['models']
        
        # Find current model index
        current_model_index = 0
        try:
            current_model_index = models.index(self.current_settings['model'])
        except ValueError:
            current_model_index = 0
        
        selected_index = current_model_index
        
        while True:
            # Display current state
            print(self.ui.clear_screen())
            print(self._create_header())
            print(self._create_model_selection_display(provider_key, self.current_settings['model'], selected_index))
            
            # Get keyboard input
            key_code, char = self.keyboard.get_key()
            
            if key_code == KeyCode.UP:
                selected_index = (selected_index - 1) % len(models)
                
            elif key_code == KeyCode.DOWN:
                selected_index = (selected_index + 1) % len(models)
                
            elif key_code == KeyCode.ENTER:
                # Select current option
                self.current_settings['model'] = models[selected_index]
                return True
                
            elif key_code == KeyCode.ESCAPE:
                return False
                
            elif char and char.lower() == 'q':
                return False

    def _configure_parameters(self) -> bool:
        """Interactive parameter configuration with arrow key navigation"""
        selected_index = 0
        
        while True:
            # Display current state
            print(self.ui.clear_screen())
            print(self._create_header())
            print(self._create_parameter_display(selected_index))
            
            # Get keyboard input
            key_code, char = self.keyboard.get_key()
            
            if key_code == KeyCode.UP:
                selected_index = (selected_index - 1) % 5  # 5 options including "skip"
                
            elif key_code == KeyCode.DOWN:
                selected_index = (selected_index + 1) % 5
                
            elif key_code == KeyCode.ENTER:
                if selected_index == 0:  # Temperature
                    self._edit_temperature()
                    continue  # Stay in parameter config
                    
                elif selected_index == 1:  # Max tokens
                    self._edit_max_tokens()
                    continue  # Stay in parameter config
                        
                elif selected_index == 2:  # API Key
                    self._edit_api_key()
                    continue  # Stay in parameter config
                    
                elif selected_index == 3:  # Host URL
                    self._edit_host_url()
                    continue  # Stay in parameter config
                        
                elif selected_index == 4:  # Skip
                    return True
                    
            elif key_code == KeyCode.ESCAPE:
                return False
                
            elif char and char.lower() == 'q':
                return False

    def run_settings_dialog(self) -> bool:
        """Run the complete settings configuration flow with arrow key navigation"""
        try:
            # Save terminal state
            print(self.ui.hide_cursor(), end='')
            
            # Step 1: Provider selection
            if not self._select_provider():
                return False
                
            # Step 2: Model selection  
            if not self._select_model():
                return False
                
            # Step 3: Parameter configuration
            if not self._configure_parameters():
                return False
                
            # Step 4: Summary and confirmation
            return self._confirm_and_save()
                    
        finally:
            # Restore terminal state
            print(self.ui.show_cursor(), end='')
            print(self.ui.clear_screen())


def run_modern_settings_dialog(theme_manager: Optional[ThemeManager] = None, ui_components: Optional[UIComponents] = None) -> bool:
    """Run the modern settings dialog"""
    settings_ui = ModernSettingsUI(theme_manager, ui_components)
    return settings_ui.run_settings_dialog()
