#!/usr/bin/env python3
"""
Configuration management module for the Human/Machine Sentence Classifier.

This module provides centralized configuration management using environment variables,
configuration files, and sensible defaults. It supports validation and type conversion.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from functools import lru_cache

try:
    from pydantic import BaseModel, Field, validator
    from pydantic_settings import BaseSettings
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback to simple dataclass if pydantic is not available
    BaseModel = object

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration values
DEFAULT_CONFIG = {
    'gemini': {
        'api_key_file': '~/.api-gemini',
        'model_file': '~/.model-gemini',
        'default_model': 'gemini-2.5-pro',
        'max_retries': 3,
        'timeout': 60
    },
    'openrouter': {
        'api_key_file': '~/.api-openrouter',
        'model_file': '~/.model-openrouter',
        'default_model': 'deepseek/deepseek-r1-0528:free',
        'max_retries': 3,
        'timeout': 60
    },
    'evaluation': {
        'default_delay': 0.1,
        'max_concurrent_requests': 1,
        'output_format': 'json'
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(levelname)s - %(message)s'
    }
}

class AppConfig(BaseModel if PYDANTIC_AVAILABLE else object):
    """Application configuration with validation."""

    # Gemini settings
    gemini_api_key: Optional[str] = Field(None, env='GEMINI_API_KEY')
    google_api_key: Optional[str] = Field(None, env='GOOGLE_API_KEY')
    gemini_model: Optional[str] = Field(None, env='GEMINI_MODEL')
    gemini_max_retries: int = Field(3, env='GEMINI_MAX_RETRIES')

    # OpenRouter settings
    openrouter_api_key: Optional[str] = Field(None, env='OPENROUTER_API_KEY')
    openrouter_model: Optional[str] = Field(None, env='OPENROUTER_MODEL')
    openrouter_max_retries: int = Field(3, env='OPENROUTER_MAX_RETRIES')

    # Evaluation settings
    default_provider: str = Field('openrouter', env='DEFAULT_PROVIDER')
    evaluation_delay: float = Field(0.1, env='EVALUATION_DELAY')
    output_dir: str = Field('results', env='OUTPUT_DIR')

    # Logging settings
    log_level: str = Field('INFO', env='LOG_LEVEL')

    if PYDANTIC_AVAILABLE:
        @validator('default_provider')
        def validate_provider(cls, v):
            if v not in ['gemini', 'openrouter']:
                raise ValueError('Provider must be either "gemini" or "openrouter"')
            return v

        @validator('evaluation_delay')
        def validate_delay(cls, v):
            if v < 0:
                raise ValueError('Evaluation delay must be non-negative')
            return v

    @property
    def effective_gemini_api_key(self) -> Optional[str]:
        """Get the effective Gemini API key from env vars or files."""
        # Try environment variables first
        if self.gemini_api_key:
            return self.gemini_api_key
        if self.google_api_key:
            return self.google_api_key

        # Try file-based configuration
        return _read_text_file(Path(DEFAULT_CONFIG['gemini']['api_key_file']).expanduser())

    @property
    def effective_openrouter_api_key(self) -> Optional[str]:
        """Get the effective OpenRouter API key from env vars or files."""
        if self.openrouter_api_key:
            return self.openrouter_api_key

        return _read_text_file(Path(DEFAULT_CONFIG['openrouter']['api_key_file']).expanduser())

    @property
    def effective_gemini_model(self) -> str:
        """Get the effective Gemini model name."""
        if self.gemini_model:
            return self.gemini_model

        model_file = Path(DEFAULT_CONFIG['gemini']['model_file']).expanduser()
        model = _read_text_file(model_file)
        return model if model else DEFAULT_CONFIG['gemini']['default_model']

    @property
    def effective_openrouter_model(self) -> str:
        """Get the effective OpenRouter model name."""
        if self.openrouter_model:
            return self.openrouter_model

        model_file = Path(DEFAULT_CONFIG['openrouter']['model_file']).expanduser()
        model = _read_text_file(model_file)
        return model if model else DEFAULT_CONFIG['openrouter']['default_model']

def _read_text_file(path: Path) -> Optional[str]:
    """Read and strip text from a file path if it exists; return None on failure or empty."""
    try:
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8").strip()
        return content or None
    except (IOError, OSError, UnicodeDecodeError) as e:
        logging.debug(f"Failed to read file {path}: {e}")
        return None

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get the application configuration with caching."""
    if PYDANTIC_AVAILABLE:
        # Use Pydantic settings if available
        class Settings(BaseSettings):
            gemini_api_key: Optional[str] = Field(None, env='GEMINI_API_KEY')
            google_api_key: Optional[str] = Field(None, env='GOOGLE_API_KEY')
            gemini_model: Optional[str] = Field(None, env='GEMINI_MODEL')
            gemini_max_retries: int = Field(3, env='GEMINI_MAX_RETRIES')
            openrouter_api_key: Optional[str] = Field(None, env='OPENROUTER_API_KEY')
            openrouter_model: Optional[str] = Field(None, env='OPENROUTER_MODEL')
            openrouter_max_retries: int = Field(3, env='OPENROUTER_MAX_RETRIES')
            default_provider: str = Field('openrouter', env='DEFAULT_PROVIDER')
            evaluation_delay: float = Field(0.1, env='EVALUATION_DELAY')
            output_dir: str = Field('results', env='OUTPUT_DIR')
            log_level: str = Field('INFO', env='LOG_LEVEL')

            class Config:
                env_file = '.env'
                env_file_encoding = 'utf-8'

        return Settings()
    else:
        # Fallback to manual configuration loading
        return AppConfig(
            gemini_api_key=os.getenv('GEMINI_API_KEY'),
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            gemini_model=os.getenv('GEMINI_MODEL'),
            gemini_max_retries=int(os.getenv('GEMINI_MAX_RETRIES', '3')),
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
            openrouter_model=os.getenv('OPENROUTER_MODEL'),
            openrouter_max_retries=int(os.getenv('OPENROUTER_MAX_RETRIES', '3')),
            default_provider=os.getenv('DEFAULT_PROVIDER', 'openrouter'),
            evaluation_delay=float(os.getenv('EVALUATION_DELAY', '0.1')),
            output_dir=os.getenv('OUTPUT_DIR', 'results'),
            log_level=os.getenv('LOG_LEVEL', 'INFO')
        )

def validate_configuration() -> Dict[str, Any]:
    """Validate the current configuration and return validation results."""
    config = get_config()
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'config_summary': {}
    }

    # Check API keys
    if not config.effective_gemini_api_key:
        results['warnings'].append("No Gemini API key found. Gemini classification will not work.")
    else:
        results['config_summary']['gemini_api_key'] = "Set"

    if not config.effective_openrouter_api_key:
        results['warnings'].append("No OpenRouter API key found. OpenRouter classification will not work.")
    else:
        results['config_summary']['openrouter_api_key'] = "Set"

    # Check models
    results['config_summary']['gemini_model'] = config.effective_gemini_model
    results['config_summary']['openrouter_model'] = config.effective_openrouter_model
    results['config_summary']['default_provider'] = config.default_provider
    results['config_summary']['evaluation_delay'] = config.evaluation_delay

    # Validate provider
    if config.default_provider not in ['gemini', 'openrouter']:
        results['errors'].append(f"Invalid default provider: {config.default_provider}")
        results['valid'] = False

    # Validate delay
    if config.evaluation_delay < 0:
        results['errors'].append(f"Invalid evaluation delay: {config.evaluation_delay}")
        results['valid'] = False

    return results

def print_config_summary():
    """Print a summary of the current configuration."""
    validation = validate_configuration()

    print("Configuration Summary:")
    print("=" * 30)

    if validation['config_summary']:
        print("Settings:")
        for key, value in validation['config_summary'].items():
            print(f"  {key}: {value}")

    if validation['errors']:
        print("\nErrors:")
        for error in validation['errors']:
            print(f"  ❌ {error}")

    if validation['warnings']:
        print("\nWarnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")

    if validation['valid']:
        print("\n✅ Configuration is valid.")
    else:
        print("\n❌ Configuration has errors.")

if __name__ == "__main__":
    print_config_summary()