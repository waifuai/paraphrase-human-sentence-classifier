#!/usr/bin/env python3
"""
Model interaction module for the Human/Machine Sentence Classifier using Google Gemini API.
"""

import os
import logging
from pathlib import Path
from typing import Optional

# Attempt to import google.generativeai and handle potential import error
try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai library not found.")
    print("Please install it using: pip install google-generativeai")
    genai = None # Set to None to handle gracefully later

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
API_KEY_FILE_PATH = Path.home() / ".api-gemini" # Path to the API key file in home directory

# --- Global variable for the initialized model ---
_model = None
_api_key_loaded = False

def _load_api_key() -> Optional[str]:
    """Loads the Gemini API key from ~/.api-gemini."""
    try:
        if API_KEY_FILE_PATH.is_file():
            logging.info(f"Reading API key from {API_KEY_FILE_PATH}")
            return API_KEY_FILE_PATH.read_text().strip()
        else:
            logging.error(f"API key file not found at: {API_KEY_FILE_PATH}")
            return None
    except Exception as e:
        logging.error(f"Failed to read API key from {API_KEY_FILE_PATH}: {e}")
        return None

def _initialize_model(model_name: str = DEFAULT_GEMINI_MODEL) -> bool:
    """Initializes the Gemini model if not already done."""
    global _model, _api_key_loaded
    if _model:
        return True # Already initialized

    if not genai:
        logging.error("google-generativeai library is not available. Cannot initialize model.")
        return False

    if not _api_key_loaded:
        api_key = _load_api_key()
        if not api_key:
            logging.error("Gemini API key could not be loaded. Cannot initialize model.")
            return False
        try:
            genai.configure(api_key=api_key)
            _api_key_loaded = True
            logging.info("Gemini API key configured successfully.")
        except Exception as e:
            logging.error(f"Failed to configure Gemini API key: {e}")
            return False

    try:
        logging.info(f"Initializing Gemini model: {model_name}")
        _model = genai.GenerativeModel(model_name)
        logging.info("Gemini model initialized successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model '{model_name}': {e}")
        _model = None # Ensure model is None if initialization fails
        return False

def classify_with_gemini(text: str, model_name: str = DEFAULT_GEMINI_MODEL) -> Optional[str]:
    """
    Classifies the input text as human-written ('1') or machine-generated ('0') using Gemini.

    Args:
        text: The text sentence to classify.
        model_name: The specific Gemini model to use.

    Returns:
        '1' if classified as human, '0' if classified as machine, or None on error.
    """
    global _model
    if not _model:
        if not _initialize_model(model_name):
            return None # Initialization failed

    # Define the prompt for binary classification
    prompt = f"""Classify the following sentence based on whether it sounds like it was written by a human or generated by a machine (like an AI paraphrase tool).

Categories:
- '1': Human-written
- '0': Machine-generated

Respond with ONLY the digit '0' or '1'. Do not include any other text, explanation, or punctuation.

Sentence: "{text}"

Classification:"""

    try:
        logging.debug(f"Sending text to Gemini for classification: '{text[:50]}...'")
        # Set safety settings to block none, as classification prompts are unlikely to trigger them
        # and we want the raw 0/1 output. Adjust if needed for different use cases.
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = _model.generate_content(prompt, safety_settings=safety_settings)

        if response.candidates:
            # Access the text from the first part of the first candidate
            generated_text = response.candidates[0].content.parts[0].text.strip()
            logging.debug(f"Gemini raw response: '{generated_text}'")

            if generated_text in ['0', '1']:
                logging.info(f"Successfully classified text. Result: {generated_text}")
                return generated_text
            else:
                # Log unexpected output
                logging.warning(f"Gemini returned unexpected output: '{generated_text}'. Expected '0' or '1'. Text was: '{text[:100]}...'")
                # Optionally, try to parse '0' or '1' if they appear in the string
                if '0' in generated_text and '1' not in generated_text: return '0'
                if '1' in generated_text and '0' not in generated_text: return '1'
                return None # Return None if output is ambiguous or incorrect format
        else:
            # Log if no candidates are returned (might indicate blocked content despite settings)
            logging.warning(f"Gemini response had no candidates. Prompt might have been blocked. Prompt: {prompt}")
            # You can inspect response.prompt_feedback here for block reasons
            # print(response.prompt_feedback)
            return None

    except Exception as e:
        logging.error(f"Error during Gemini API call for text '{text[:50]}...': {e}")
        # Handle specific exceptions if needed (e.g., rate limits, connection errors)
        return None

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Ensure you have a ~/.api-gemini file with your key for this test
    print("--- Testing Gemini Classifier ---")
    if not API_KEY_FILE_PATH.is_file():
        print(f"WARNING: API key file not found at {API_KEY_FILE_PATH}. Create it to run this test.")
        test_text_human = None # Skip tests if no key
    else:
        test_text_human = "This is a sentence written by a real person, expressing a genuine thought."
        test_text_machine = "Pursuant to the aforementioned communication, the requested data has been compiled."
        test_text_ambiguous = "The sky is blue."

        print(f"\nClassifying Human Text: '{test_text_human}'")
        result_human = classify_with_gemini(test_text_human)
        print(f"Result: {result_human}")

        print(f"\nClassifying Machine Text: '{test_text_machine}'")
        result_machine = classify_with_gemini(test_text_machine)
        print(f"Result: {result_machine}")

        print(f"\nClassifying Ambiguous Text: '{test_text_ambiguous}'")
        result_ambiguous = classify_with_gemini(test_text_ambiguous)
        print(f"Result: {result_ambiguous}")

    print("\n--- Test Complete ---")