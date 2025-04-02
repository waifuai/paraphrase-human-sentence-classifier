import os
import pytest
import tempfile
import numpy as np
from pathlib import Path
from datasets import DatasetDict, load_dataset

# Import functions/classes from the refactored code
# Adjust imports based on the new structure (src/classifier)
from classifier.data import load_and_tokenize_data
from classifier.model import load_model
# Assuming evaluate script is now src/scripts/evaluate.py
# Need to ensure PYTHONPATH allows importing scripts directly or adjust import
# For simplicity here, let's assume compute_metrics is accessible
# If running pytest with `python -m pytest`, PYTHONPATH in pytest.ini should handle this
from scripts.evaluate import compute_metrics

# Use a smaller model for faster testing
TEST_MODEL_NAME = "prajjwal1/bert-tiny"
# TEST_MODEL_NAME = "distilbert-base-uncased" # Can switch for more thorough testing

# -------------------------
# Fixtures for Test Data
# -------------------------

@pytest.fixture(scope="module") # Use module scope for efficiency
def tsv_files(tmp_path_factory):
    # Create temporary TSV files for training and evaluation in a module-scoped temp dir
    tmp_dir = tmp_path_factory.mktemp("data")
    train_file = tmp_dir / "train.tsv"
    eval_file = tmp_dir / "eval.tsv"

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
    return str(train_file), str(eval_file)

# -------------------------
# Tests for data.py
# -------------------------

def test_load_and_tokenize_data(tsv_files):
    train_file, eval_file = tsv_files
    tokenized_datasets = load_and_tokenize_data(
        train_file=train_file,
        eval_file=eval_file,
        tokenizer_name=TEST_MODEL_NAME,
        max_length=32 # Use smaller max_length for testing
    )

    assert isinstance(tokenized_datasets, DatasetDict)
    assert "train" in tokenized_datasets
    assert "test" in tokenized_datasets
    assert len(tokenized_datasets["train"]) == 4
    assert len(tokenized_datasets["test"]) == 2

    # Check features of the tokenized dataset
    # Features depend on the tokenizer; bert-tiny doesn't add token_type_ids by default
    expected_features = ['labels', 'input_ids', 'attention_mask']
    # expected_features_alt = ['labels', 'input_ids', 'token_type_ids', 'attention_mask'] # For models like BERT base

    train_features = list(tokenized_datasets["train"].features.keys())
    test_features = list(tokenized_datasets["test"].features.keys())

    # Check if all expected features are present
    assert all(f in train_features for f in expected_features)
    assert all(f in test_features for f in expected_features)

    # Check content of one sample
    sample = tokenized_datasets["train"][0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    # After set_format("torch"), data might be Tensors, check type or convert back if needed for simple checks
    # assert isinstance(sample["input_ids"], list) # This might fail if formatted as tensor
    assert sample["labels"] == 1 # First training sample has label 1

def test_load_and_tokenize_data_split(tsv_files):
    train_file, _ = tsv_files # Only need train file for splitting
    tokenized_datasets = load_and_tokenize_data(
        train_file=train_file,
        eval_file=None, # Trigger split
        tokenizer_name=TEST_MODEL_NAME,
        max_length=32,
        test_size=0.5 # Split 4 samples into 2 train, 2 test
    )

    assert isinstance(tokenized_datasets, DatasetDict)
    assert "train" in tokenized_datasets
    assert "test" in tokenized_datasets
    assert len(tokenized_datasets["train"]) == 2
    assert len(tokenized_datasets["test"]) == 2

# -------------------------
# Tests for model.py
# -------------------------

def test_load_model():
    model = load_model(model_name=TEST_MODEL_NAME, num_labels=2)
    assert model is not None
    assert model.config.num_labels == 2
    # Check if it's actually a sequence classification model (has the right head)
    assert hasattr(model, "classifier") or hasattr(model, "score") # Different models name the head differently

    model_3_labels = load_model(model_name=TEST_MODEL_NAME, num_labels=3)
    assert model_3_labels.config.num_labels == 3

# -------------------------
# Tests for evaluate.py (compute_metrics)
# -------------------------

def test_compute_metrics():
    # Mock EvalPrediction object
    class MockEvalPrediction:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids

    # Perfect prediction case
    preds = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1]]) # Logits or probabilities
    labels = np.array([1, 0, 1, 0])
    eval_pred = MockEvalPrediction(predictions=preds, label_ids=labels)
    metrics = compute_metrics(eval_pred)

    assert metrics['accuracy'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['confusion_matrix'] == [[2, 0], [0, 2]] # [[TN, FP], [FN, TP]]

    # Imperfect prediction case
    preds_imperfect = np.array([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6]]) # Logits or probabilities
    # Predictions:             1            0            0            1
    labels_imperfect = np.array([1,           0,           1,           0])
    # Actual:                  1            0            1            0
    # Results:                 TP           TN           FN           FP
    eval_pred_imperfect = MockEvalPrediction(predictions=preds_imperfect, label_ids=labels_imperfect)
    metrics_imperfect = compute_metrics(eval_pred_imperfect)

    assert metrics_imperfect['accuracy'] == 0.5 # TP + TN / Total = 1 + 1 / 4 = 0.5
    assert metrics_imperfect['f1'] == pytest.approx(0.5) # 2 * (P*R)/(P+R) = 2 * (0.5*0.5)/(0.5+0.5) = 2 * 0.25 / 1 = 0.5
    assert metrics_imperfect['precision'] == 0.5 # TP / (TP + FP) = 1 / (1 + 1) = 0.5
    assert metrics_imperfect['recall'] == 0.5 # TP / (TP + FN) = 1 / (1 + 1) = 0.5
    assert metrics_imperfect['confusion_matrix'] == [[1, 1], [1, 1]] # [[TN, FP], [FN, TP]]

    # Case with zero division
    preds_zero = np.array([[0.8, 0.2], [0.7, 0.3]]) # Predicts all 0
    labels_zero = np.array([0, 0]) # Actual all 0
    eval_pred_zero = MockEvalPrediction(predictions=preds_zero, label_ids=labels_zero)
    metrics_zero = compute_metrics(eval_pred_zero)
    assert metrics_zero['accuracy'] == 1.0
    assert metrics_zero['f1'] == 0.0 # F1 is 0 if precision or recall is 0
    assert metrics_zero['precision'] == 0.0 # TP / (TP + FP) = 0 / (0 + 0) -> 0 by zero_division=0
    assert metrics_zero['recall'] == 0.0 # TP / (TP + FN) = 0 / (0 + 0) -> 0 by zero_division=0
    assert metrics_zero['confusion_matrix'] == [[2, 0], [0, 0]] # [[TN, FP], [FN, TP]]

    # Case with only one class predicted/present (all positive)
    preds_one_class = np.array([[0.1, 0.9], [0.2, 0.8]]) # Predicts all 1
    labels_one_class = np.array([1, 1]) # Actual all 1
    eval_pred_one_class = MockEvalPrediction(predictions=preds_one_class, label_ids=labels_one_class)
    metrics_one_class = compute_metrics(eval_pred_one_class)
    assert metrics_one_class['accuracy'] == 1.0
    assert metrics_one_class['f1'] == 1.0
    assert metrics_one_class['precision'] == 1.0
    assert metrics_one_class['recall'] == 1.0
    assert metrics_one_class['confusion_matrix'] == [[0, 0], [0, 2]] # [[TN, FP], [FN, TP]]
