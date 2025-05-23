import os
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Import functions/classes from the refactored code
# Adjust imports based on the new structure (src/classifier)
# Need to ensure PYTHONPATH allows importing scripts directly or adjust import
# For simplicity here, let's assume compute_metrics is accessible
# If running pytest with `python -m pytest`, PYTHONPATH in pytest.ini should handle this
# Note: Running tests might require setting PYTHONPATH=. or using `python -m pytest`
try:
    from classifier.data import load_data_from_tsv
    from classifier.model import classify_with_gemini, _initialize_model, _model, _api_key_loaded, API_KEY_FILE_PATH
    from scripts.evaluate import compute_metrics
except ImportError as e:
    # Handle cases where running pytest directly might cause import issues
    # This is a common issue depending on how pytest discovers tests vs how modules are run
    print(f"Import Error: {e}. Ensure PYTHONPATH includes 'src' or run with 'python -m pytest'.")
    # Define dummy functions/classes to allow tests to be collected even if imports fail initially
    def load_data_from_tsv(*args, **kwargs): return {'train': [], 'test': []}
    def classify_with_gemini(*args, **kwargs): return None
    def compute_metrics(*args, **kwargs): return {}
    _model = None
    _api_key_loaded = False
    API_KEY_FILE_PATH = Path.home() / ".api-gemini"
    def _initialize_model(*args, **kwargs): return False


# -------------------------
# Fixtures for Test Data
# -------------------------

@pytest.fixture
def tsv_files(tmp_path):
    # Create temporary TSV files for training and evaluation
    train_file = tmp_path / "train.tsv"
    eval_file = tmp_path / "eval.tsv"
    empty_file = tmp_path / "empty.tsv"
    malformed_file = tmp_path / "malformed.tsv"

    # New format: text\tlabel
    train_file.write_text(
        "Human sentence 1\t1\n"
        "Machine sentence 1\t0\n"
        "Another human sentence\t1\n"
        "Another machine sentence\t0\n"
    )
    eval_file.write_text(
        "Evaluation human sentence\t1\n"
        "Evaluation machine sentence\t0\n"
    )
    empty_file.write_text("")
    malformed_file.write_text(
        "Good line\t1\n"
        "Bad label\tX\n"
        "Too few columns\n"
        "Too many columns\t0\tExtra\n"
        "Good line 2\t0\n"
    )
    return str(train_file), str(eval_file), str(empty_file), str(malformed_file)

# -------------------------
# Tests for data.py
# -------------------------

def test_load_data_from_tsv_both_files(tsv_files):
    train_file, eval_file, _, _ = tsv_files
    datasets = load_data_from_tsv(train_file=train_file, eval_file=eval_file)

    assert "train" in datasets
    assert "test" in datasets
    assert len(datasets["train"]) == 4
    assert len(datasets["test"]) == 2
    assert datasets["train"][0] == {"text": "Human sentence 1", "label": 1}
    assert datasets["test"][1] == {"text": "Evaluation machine sentence", "label": 0}

def test_load_data_from_tsv_split(tsv_files):
    train_file, _, _, _ = tsv_files
    datasets = load_data_from_tsv(train_file=train_file, eval_file=None, test_size=0.5, random_seed=123)

    assert "train" in datasets
    assert "test" in datasets
    # Lengths depend on shuffling, but should sum to 4
    assert len(datasets["train"]) + len(datasets["test"]) == 4
    # Check if split occurred (lengths should likely be 2, 2 unless edge case)
    assert len(datasets["train"]) > 0
    assert len(datasets["test"]) > 0


def test_load_data_from_tsv_eval_only(tsv_files):
    _, eval_file, _, _ = tsv_files
    datasets = load_data_from_tsv(train_file=None, eval_file=eval_file)

    assert "train" in datasets
    assert "test" in datasets
    assert len(datasets["train"]) == 0
    assert len(datasets["test"]) == 2
    assert datasets["test"][0] == {"text": "Evaluation human sentence", "label": 1}

def test_load_data_from_tsv_malformed(tsv_files):
    _, _, _, malformed_file = tsv_files
    datasets = load_data_from_tsv(train_file=malformed_file, eval_file=None)
    # Should load only the 2 good lines
    assert len(datasets["train"]) == 2
    assert datasets["train"][0] == {"text": "Good line", "label": 1}
    assert datasets["train"][1] == {"text": "Good line 2", "label": 0}

def test_load_data_from_tsv_empty(tsv_files):
     _, _, empty_file, _ = tsv_files
     datasets = load_data_from_tsv(train_file=empty_file, eval_file=None)
     assert len(datasets["train"]) == 0
     assert len(datasets["test"]) == 0 # Split results in empty test set

def test_load_data_from_tsv_not_found():
    with pytest.raises(FileNotFoundError):
        load_data_from_tsv(train_file="nonexistent_file.tsv", eval_file=None)

# -------------------------
# Tests for model.py
# -------------------------

# Reset global state before each model test
@pytest.fixture(autouse=True)
def reset_model_state():
    global _model, _api_key_loaded
    _model = None
    _api_key_loaded = False

# Mock Path.home() and Path.is_file() / read_text() for API key tests
@patch('classifier.model.Path.home', return_value=Path('/fake/home'))
def test_initialize_model_key_found(mock_home):
    # Mock the specific file path check and read
    with patch('pathlib.Path.is_file', return_value=True), \
         patch('pathlib.Path.read_text', return_value='fake-api-key'):
        # Mock the genai configure and GenerativeModel
        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_gen_model:
            assert _initialize_model() is True
            mock_configure.assert_called_once_with(api_key='fake-api-key')
            mock_gen_model.assert_called_once()
            assert _api_key_loaded is True
            assert _model is not None

@patch('classifier.model.Path.home', return_value=Path('/fake/home'))
def test_initialize_model_key_not_found(mock_home):
     with patch('pathlib.Path.is_file', return_value=False):
         with patch('google.generativeai.configure') as mock_configure, \
              patch('google.generativeai.GenerativeModel') as mock_gen_model:
            assert _initialize_model() is False
            mock_configure.assert_not_called()
            mock_gen_model.assert_not_called()
            assert _api_key_loaded is False
            assert _model is None

@patch('classifier.model.Path.home', return_value=Path('/fake/home'))
@patch('pathlib.Path.is_file', return_value=True)
@patch('pathlib.Path.read_text', return_value='fake-api-key')
@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
def test_classify_with_gemini_success(mock_gen_model, mock_configure, mock_read, mock_is_file, mock_home):
    # Mock the response from generate_content
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_part = MagicMock()
    mock_part.text = '1' # Simulate Gemini returning '1'
    mock_candidate.content.parts = [mock_part]
    mock_response.candidates = [mock_candidate]
    # Configure the mock model instance returned by GenerativeModel()
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_gen_model.return_value = mock_model_instance

    # Initialize first (mocks will be used)
    assert _initialize_model() is True

    # Now classify
    result = classify_with_gemini("This is a test")
    assert result == '1'
    mock_model_instance.generate_content.assert_called_once()
    # Check prompt contains the text (optional, but good)
    call_args, _ = mock_model_instance.generate_content.call_args
    assert "This is a test" in call_args[0] # Prompt is the first arg

@patch('classifier.model.Path.home', return_value=Path('/fake/home'))
@patch('pathlib.Path.is_file', return_value=True)
@patch('pathlib.Path.read_text', return_value='fake-api-key')
@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
def test_classify_with_gemini_unexpected_output(mock_gen_model, mock_configure, mock_read, mock_is_file, mock_home):
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_part = MagicMock()
    mock_part.text = ' The classification is 0 ' # Simulate extra text
    mock_candidate.content.parts = [mock_part]
    mock_response.candidates = [mock_candidate]
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_gen_model.return_value = mock_model_instance

    assert _initialize_model() is True
    result = classify_with_gemini("Another test")
    # The current logic tries to parse '0' or '1' if present
    assert result == '0' # It should find '0' in the string

@patch('classifier.model.Path.home', return_value=Path('/fake/home'))
@patch('pathlib.Path.is_file', return_value=True)
@patch('pathlib.Path.read_text', return_value='fake-api-key')
@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
def test_classify_with_gemini_api_error(mock_gen_model, mock_configure, mock_read, mock_is_file, mock_home):
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.side_effect = Exception("API rate limit exceeded")
    mock_gen_model.return_value = mock_model_instance

    assert _initialize_model() is True
    result = classify_with_gemini("Error test")
    assert result is None

@patch('classifier.model.Path.home', return_value=Path('/fake/home'))
@patch('pathlib.Path.is_file', return_value=True)
@patch('pathlib.Path.read_text', return_value='fake-api-key')
@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
def test_classify_with_gemini_no_candidates(mock_gen_model, mock_configure, mock_read, mock_is_file, mock_home):
    mock_response = MagicMock()
    mock_response.candidates = [] # Simulate no candidates (e.g., blocked prompt)
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_gen_model.return_value = mock_model_instance

    assert _initialize_model() is True
    result = classify_with_gemini("Blocked content test")
    assert result is None

# -------------------------
# Tests for evaluate.py (compute_metrics) - Kept from original
# -------------------------

def test_compute_metrics():
    # Perfect prediction case
    labels = [1, 0, 1, 0]
    preds = [1, 0, 1, 0]
    metrics = compute_metrics(labels, preds)

    assert metrics['accuracy'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['confusion_matrix'] == [[2, 0], [0, 2]] # [[TN, FP], [FN, TP]]
    assert metrics['support'] == 4

    # Imperfect prediction case
    labels_imperfect = [1, 0, 1, 0]
    # Actual:           1  0  1  0
    preds_imperfect =  [1, 0, 0, 1]
    # Predicted:        1  0  0  1
    # Results:          TP TN FN FP
    metrics_imperfect = compute_metrics(labels_imperfect, preds_imperfect)

    assert metrics_imperfect['accuracy'] == 0.5 # TP + TN / Total = 1 + 1 / 4 = 0.5
    assert metrics_imperfect['f1'] == pytest.approx(0.5) # 2 * (P*R)/(P+R) = 2 * (0.5*0.5)/(0.5+0.5) = 0.5
    assert metrics_imperfect['precision'] == 0.5 # TP / (TP + FP) = 1 / (1 + 1) = 0.5
    assert metrics_imperfect['recall'] == 0.5 # TP / (TP + FN) = 1 / (1 + 1) = 0.5
    assert metrics_imperfect['confusion_matrix'] == [[1, 1], [1, 1]] # [[TN, FP], [FN, TP]]
    assert metrics_imperfect['support'] == 4

    # Case with zero division (predicts all 0, actual all 0)
    labels_zero = [0, 0]
    preds_zero = [0, 0]
    metrics_zero = compute_metrics(labels_zero, preds_zero)
    assert metrics_zero['accuracy'] == 1.0
    # F1/Precision/Recall for binary average with only negatives is tricky.
    # sklearn gives 0 by default with zero_division=0
    assert metrics_zero['f1'] == 0.0
    assert metrics_zero['precision'] == 0.0
    assert metrics_zero['recall'] == 0.0
    assert metrics_zero['confusion_matrix'] == [[2, 0], [0, 0]] # [[TN, FP], [FN, TP]]
    assert metrics_zero['support'] == 2

    # Case with only one class predicted/present (all positive)
    labels_one_class = [1, 1]
    preds_one_class = [1, 1]
    metrics_one_class = compute_metrics(labels_one_class, preds_one_class)
    assert metrics_one_class['accuracy'] == 1.0
    assert metrics_one_class['f1'] == 1.0
    assert metrics_one_class['precision'] == 1.0
    assert metrics_one_class['recall'] == 1.0
    assert metrics_one_class['confusion_matrix'] == [[0, 0], [0, 2]] # [[TN, FP], [FN, TP]]
    assert metrics_one_class['support'] == 2
