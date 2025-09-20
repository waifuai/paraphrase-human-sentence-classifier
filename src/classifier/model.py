#!/usr/bin/env python3
"""
Model interaction module for the Human/Machine Sentence Classifier using Google GenAI SDK and OpenRouter.

This module provides classification functionality for determining whether text is human-written or machine-generated
using various AI providers with robust error handling, caching, and retry mechanisms.
"""

import os
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Import configuration
try:
    from .config import get_config, AppConfig
except ImportError:
    # Fallback if config module is not available
    get_config = None

# Import caching
try:
    from .cache import get_cache, ResponseCache
except ImportError:
    # Fallback if cache module is not available
    get_cache = None

# New SDK import
try:
    from google import genai
except Exception as e:
    genai = None

# Optional dependency for OpenRouter HTTP client
try:
    import requests
except Exception:
    requests = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Allow default model names to be overridden by user home dotfiles:
#   ~/.model-gemini        -> overrides DEFAULT_GEMINI_MODEL
#   ~/.model-openrouter    -> overrides DEFAULT_OPENROUTER_MODEL
API_KEY_FILE_PATH = Path.home() / ".api-gemini"  # Path to the API key file in home directory
GEMINI_MODEL_FILE_PATH = Path.home() / ".model-gemini"

# OpenRouter configuration
OPENROUTER_API_KEY_FILE_PATH = Path.home() / ".api-openrouter"
OPENROUTER_MODEL_FILE_PATH = Path.home() / ".model-openrouter"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Hardcoded fallbacks if no override files are present
_DEFAULT_GEMINI_MODEL_FALLBACK = "gemini-2.5-pro"
_DEFAULT_OPENROUTER_MODEL_FALLBACK = "deepseek/deepseek-r1-0528:free"

# --- Global state ---
_client: Optional["genai.Client"] = None

def _read_text_file(path: Path) -> Optional[str]:
    """Read and strip text from a file path if it exists; return None on failure or empty.

    Args:
        path: Path to the file to read

    Returns:
        Stripped text content or None if file doesn't exist, is empty, or can't be read
    """
    try:
        if not path.exists():
            logging.debug(f"File does not exist: {path}")
            return None

        if not path.is_file():
            logging.debug(f"Path is not a file: {path}")
            return None

        content = path.read_text(encoding="utf-8").strip()
        return content or None

    except (IOError, OSError, UnicodeDecodeError) as e:
        logging.error(f"Failed to read text from {path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error reading {path}: {e}")
        return None

def _resolve_gemini_default_model() -> str:
    """Resolve default Gemini model name from configuration system."""
    if get_config:
        config = get_config()
        return config.effective_gemini_model

    # Fallback to legacy method
    override = _read_text_file(GEMINI_MODEL_FILE_PATH)
    return override if override else _DEFAULT_GEMINI_MODEL_FALLBACK

def _resolve_openrouter_default_model() -> str:
    """Resolve default OpenRouter model name from configuration system."""
    if get_config:
        config = get_config()
        return config.effective_openrouter_model

    # Fallback to legacy method
    override = _read_text_file(OPENROUTER_MODEL_FILE_PATH)
    return override if override else _DEFAULT_OPENROUTER_MODEL_FALLBACK

# Export resolved defaults so callers (e.g., CLI) can import them
DEFAULT_GEMINI_MODEL = _resolve_gemini_default_model()
DEFAULT_OPENROUTER_MODEL = _resolve_openrouter_default_model()

@dataclass
class ClassificationResult:
    """Result of a text classification attempt."""
    label: Optional[str] = None  # '0', '1', or None for failure
    confidence: Optional[float] = None  # Confidence score if available
    error: Optional[str] = None  # Error message if classification failed
    provider: Optional[str] = None  # 'gemini' or 'openrouter'
    model: Optional[str] = None  # Model name used
    response_time: Optional[float] = None  # Time taken for classification

    @property
    def is_success(self) -> bool:
        """Check if classification was successful."""
        return self.label is not None and self.error is None

    @property
    def is_human(self) -> Optional[bool]:
        """Check if text was classified as human-written."""
        if not self.is_success:
            return None
        return self.label == '1'

    @property
    def is_machine(self) -> Optional[bool]:
        """Check if text was classified as machine-generated."""
        if not self.is_success:
            return None
        return self.label == '0'

def _shared_prompt(text: str) -> str:
    """Generate the shared classification prompt used for both providers.

    Args:
        text: The text to classify

    Returns:
        Formatted prompt string for the AI model

    Raises:
        ValueError: If text is empty or None
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    # Sanitize text to prevent prompt injection
    sanitized_text = text.strip().replace("\\", "\\\\").replace("\"", "\\\"")

    return (
        "Classify the following sentence based on whether it sounds like it was "
        "written by a human or generated by a machine like an AI paraphrase tool.\n\n"
        "Categories:\n"
        "- '1': Human-written\n"
        "- '0': Machine-generated\n\n"
        "Respond with ONLY the digit '0' or '1'. Do not include any other text, "
        "explanation, or punctuation.\n\n"
        f"Sentence: \"{sanitized_text}\"\n\n"
        "Classification:"
    )

# ---------------- Gemini support ----------------

def _read_key_file() -> Optional[str]:
    """Read API key from ~/.api-gemini if present."""
    try:
        if API_KEY_FILE_PATH.is_file():
            logging.info(f"Reading API key from {API_KEY_FILE_PATH}")
            return API_KEY_FILE_PATH.read_text(encoding="utf-8").strip()
        return None
    except Exception as e:
        logging.error(f"Failed to read API key from {API_KEY_FILE_PATH}: {e}")
        return None

def _resolve_api_key() -> Optional[str]:
    """Resolve API key using configuration system.

    Returns:
        API key from configuration or None if not available.
    """
    if get_config:
        config = get_config()
        return config.effective_gemini_api_key

    # Fallback to legacy method if config not available
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return _read_key_file()

def _ensure_client() -> bool:
    """Initialize a singleton genai.Client."""
    global _client
    if _client is not None:
        return True
    if genai is None:
        logging.error("google-genai SDK not available. Install with: google-genai~=1.28")
        return False
    api_key = _resolve_api_key()
    if not api_key:
        # For tests that patch genai.Client, we still return False when no key
        logging.error("No API key found in GEMINI_API_KEY / GOOGLE_API_KEY or ~/.api-gemini")
        return False
    try:
        _client = genai.Client(api_key=api_key)
        logging.info("Google GenAI client initialized.")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize Google GenAI client: {e}")
        _client = None
        return False

def classify_with_gemini(
    text: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
    max_retries: int = 3,
    use_cache: bool = True
) -> ClassificationResult:
    """
    Classifies the input text as human-written ('1') or machine-generated ('0') using Gemini.

    Args:
        text: The text to classify
        model_name: Name of the Gemini model to use
        max_retries: Maximum number of retry attempts on failure
        use_cache: Whether to use caching for responses

    Returns:
        ClassificationResult with the classification outcome
    """
    result = ClassificationResult(provider="gemini", model=model_name)

    # Input validation
    if not text or not text.strip():
        result.error = "Text cannot be empty"
        return result

    # Check cache first
    if use_cache and get_cache:
        cache = get_cache()
        cached_result = cache.get(text, "gemini", model_name)
        if cached_result is not None:
            logging.debug("Using cached Gemini result")
            result.label = cached_result
            result.response_time = 0.0  # Cached result has no API latency
            return result

    start_time = time.time()

    if not _ensure_client():
        result.error = "Gemini client not available"
        return result

    for attempt in range(max_retries):
        try:
            prompt = _shared_prompt(text)

            # New SDK call style: client.models.generate_content(...)
            resp = _client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            # Extract text content with multiple fallback strategies
            generated_text = None

            # Strategy 1: Direct text attribute
            try:
                if hasattr(resp, "text") and isinstance(resp.text, str):
                    generated_text = resp.text.strip()
            except Exception:
                pass

            # Strategy 2: Candidates structure
            if generated_text is None:
                try:
                    cand = getattr(resp, "candidates", None)
                    if cand and len(cand) > 0:
                        content = getattr(cand[0], "content", None)
                        parts = getattr(content, "parts", None)
                        if parts and len(parts) > 0 and hasattr(parts[0], "text"):
                            generated_text = parts[0].text.strip()
                except Exception:
                    pass

            # Strategy 3: String representation as last resort
            if generated_text is None:
                try:
                    generated_text = str(resp).strip()
                except Exception:
                    pass

            if not generated_text:
                error_msg = "Gemini response contained no text content"
                if attempt < max_retries - 1:
                    logging.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    result.error = error_msg
                    return result

            # Parse and validate response
            if generated_text in ("0", "1"):
                result.label = generated_text
                result.response_time = time.time() - start_time

                # Cache successful result
                if use_cache and get_cache:
                    cache = get_cache()
                    cache.put(text, "gemini", model_name, result.label,
                            metadata={"response_time": result.response_time})

                return result

            # Heuristic fallback for ambiguous responses
            if "0" in generated_text and "1" not in generated_text:
                result.label = "0"
                result.response_time = time.time() - start_time
                return result
            elif "1" in generated_text and "0" not in generated_text:
                result.label = "1"
                result.response_time = time.time() - start_time
                return result

            # If we can't parse the response
            error_msg = f"Unexpected Gemini output: '{generated_text[:100]}...'. Expected '0' or '1'."
            if attempt < max_retries - 1:
                logging.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            else:
                result.error = error_msg
                return result

        except Exception as e:
            error_msg = f"Gemini API call failed: {e}"
            if attempt < max_retries - 1:
                logging.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            else:
                result.error = error_msg
                return result

    result.error = "Max retries exceeded"
    return result

# ---------------- OpenRouter support ----------------

def _read_openrouter_key_file() -> Optional[str]:
    """Read OpenRouter API key from ~/.api-openrouter if present."""
    try:
        if OPENROUTER_API_KEY_FILE_PATH.is_file():
            logging.info(f"Reading OpenRouter API key from {OPENROUTER_API_KEY_FILE_PATH}")
            return OPENROUTER_API_KEY_FILE_PATH.read_text(encoding="utf-8").strip()
        return None
    except Exception as e:
        logging.error(f"Failed to read OpenRouter API key from {OPENROUTER_API_KEY_FILE_PATH}: {e}")
        return None

def _resolve_openrouter_api_key() -> Optional[str]:
    """Resolve OpenRouter API key using configuration system."""
    if get_config:
        config = get_config()
        return config.effective_openrouter_api_key

    # Fallback to legacy method
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return _read_openrouter_key_file()

def classify_with_openrouter(
    text: str,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
    max_retries: int = 3,
    use_cache: bool = True
) -> ClassificationResult:
    """
    Classify using OpenRouter chat completions API. Returns ClassificationResult.

    Args:
        text: The text to classify
        model_name: Name of the OpenRouter model to use
        max_retries: Maximum number of retry attempts on failure
        use_cache: Whether to use caching for responses

    Returns:
        ClassificationResult with the classification outcome
    """
    result = ClassificationResult(provider="openrouter", model=model_name)

    # Input validation
    if not text or not text.strip():
        result.error = "Text cannot be empty"
        return result

    # Check cache first
    if use_cache and get_cache:
        cache = get_cache()
        cached_result = cache.get(text, "openrouter", model_name)
        if cached_result is not None:
            logging.debug("Using cached OpenRouter result")
            result.label = cached_result
            result.response_time = 0.0  # Cached result has no API latency
            return result

    start_time = time.time()

    if requests is None:
        result.error = "requests library not available. Install it to use OpenRouter support."
        return result

    api_key = _resolve_openrouter_api_key()
    if not api_key:
        result.error = "No OpenRouter API key found in OPENROUTER_API_KEY or ~/.api-openrouter"
        return result

    for attempt in range(max_retries):
        try:
            prompt = _shared_prompt(text)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",  # Required by OpenRouter
                "X-Title": "Human vs Machine Sentence Classifier",
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 10  # Limit response length
            }

            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            if resp.status_code != 200:
                error_msg = f"OpenRouter API error {resp.status_code}: {resp.text[:200]}"
                if attempt < max_retries - 1:
                    logging.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    result.error = error_msg
                    return result

            data = resp.json()

            # Try to extract assistant message content
            generated_text = None
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                generated_text = (msg.get("content") or "").strip()

            if not generated_text:
                error_msg = "OpenRouter response contained no text content"
                if attempt < max_retries - 1:
                    logging.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    result.error = error_msg
                    return result

            # Parse and validate response
            if generated_text in ("0", "1"):
                result.label = generated_text
                result.response_time = time.time() - start_time

                # Cache successful result
                if use_cache and get_cache:
                    cache = get_cache()
                    cache.put(text, "openrouter", model_name, result.label,
                            metadata={"response_time": result.response_time})

                return result

            # Heuristic fallback for ambiguous responses
            if "0" in generated_text and "1" not in generated_text:
                result.label = "0"
                result.response_time = time.time() - start_time

                # Cache successful result
                if use_cache and get_cache:
                    cache = get_cache()
                    cache.put(text, "openrouter", model_name, result.label,
                            metadata={"response_time": result.response_time})

                return result
            elif "1" in generated_text and "0" not in generated_text:
                result.label = "1"
                result.response_time = time.time() - start_time

                # Cache successful result
                if use_cache and get_cache:
                    cache = get_cache()
                    cache.put(text, "openrouter", model_name, result.label,
                            metadata={"response_time": result.response_time})

                return result

            # If we can't parse the response
            error_msg = f"Unexpected OpenRouter output: '{generated_text[:100]}...'. Expected '0' or '1'."
            if attempt < max_retries - 1:
                logging.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            else:
                result.error = error_msg
                return result

        except requests.exceptions.Timeout:
            error_msg = "OpenRouter API request timed out"
            if attempt < max_retries - 1:
                logging.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            else:
                result.error = error_msg
                return result

        except requests.exceptions.RequestException as e:
            error_msg = f"OpenRouter API request failed: {e}"
            if attempt < max_retries - 1:
                logging.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            else:
                result.error = error_msg
                return result

        except Exception as e:
            error_msg = f"Unexpected error during OpenRouter API call: {e}"
            if attempt < max_retries - 1:
                logging.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            else:
                result.error = error_msg
                return result

    result.error = "Max retries exceeded"
    return result

def classify_text(
    text: str,
    provider: str = "openrouter",
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
    max_retries: int = 3,
    use_cache: bool = True
) -> ClassificationResult:
    """
    Unified function to classify text using either Gemini or OpenRouter.

    Args:
        text: The text to classify
        provider: Either 'gemini' or 'openrouter'
        gemini_model: Model name for Gemini provider
        openrouter_model: Model name for OpenRouter provider
        max_retries: Maximum number of retry attempts
        use_cache: Whether to use caching for responses

    Returns:
        ClassificationResult with the classification outcome

    Raises:
        ValueError: If provider is not supported
    """
    if provider.lower() == "gemini":
        return classify_with_gemini(text, gemini_model, max_retries, use_cache)
    elif provider.lower() == "openrouter":
        return classify_with_openrouter(text, openrouter_model, max_retries, use_cache)
    else:
        result = ClassificationResult()
        result.error = f"Unsupported provider: {provider}. Use 'gemini' or 'openrouter'."
        return result

if __name__ == "__main__":
    print("--- Testing Providers ---")
    example = "This is a sentence written by a real person, expressing a genuine thought."

    # Gemini quick test
    print("Testing Gemini...")
    gemini_result = classify_with_gemini(example)
    if gemini_result.is_success:
        print(f"Gemini result: {gemini_result.label} (took {gemini_result.response_time:.2f}s)")
    else:
        print(f"Gemini failed: {gemini_result.error}")

    # OpenRouter quick test
    print("\nTesting OpenRouter...")
    openrouter_result = classify_with_openrouter(example)
    if openrouter_result.is_success:
        print(f"OpenRouter result: {openrouter_result.label} (took {openrouter_result.response_time:.2f}s)")
    else:
        print(f"OpenRouter failed: {openrouter_result.error}")

    # Unified test
    print("\nTesting unified function...")
    unified_result = classify_text(example, provider="openrouter")
    if unified_result.is_success:
        print(f"Unified result: {unified_result.label} (provider: {unified_result.provider})")
    else:
        print(f"Unified failed: {unified_result.error}")