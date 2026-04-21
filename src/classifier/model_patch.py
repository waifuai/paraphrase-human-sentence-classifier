from typing import Optional
from pathlib import Path

# Monkey-patch the Settings class to add computed properties
from pydantic_settings import BaseSettings
from pydantic import Field as PydanticField

# We'll patch this after the Settings class is defined
_original_settings_init = None

def _add_computed_properties_to_settings():
    """Add computed properties to BaseSettings for compatibility."""
    from classifier.config import Settings, _DEFAULT_OPENROUTER_MODEL_FALLBACK
    from classifier.model import _read_text_file

    # Add properties to Settings class
    @property
    def effective_openrouter_api_key(self) -> Optional[str]:
        """Get the effective OpenRouter API key from env vars or files."""
        if self.openrouter_api_key:
            return self.openrouter_api_key

        return _read_text_file(Path.home() / '.api-openrouter')

    @property
    def effective_openrouter_model(self) -> str:
        """Get the effective OpenRouter model name."""
        if self.openrouter_model:
            return self.openrouter_model

        model_file = Path.home() / ".model-openrouter"
        model = _read_text_file(model_file)
        return model if model else _DEFAULT_OPENROUTER_MODEL_FALLBACK

    # Only add if not already present
    if not hasattr(Settings, 'effective_openrouter_api_key'):
        Settings.effective_openrouter_api_key = effective_openrouter_api_key
    if not hasattr(Settings, 'effective_openrouter_model'):
        Settings.effective_openrouter_model = effective_openrouter_model

# Import and patch after Settings is defined
from classifier import config
if hasattr(config, 'Settings'):
    _add_computed_properties_to_settings()
