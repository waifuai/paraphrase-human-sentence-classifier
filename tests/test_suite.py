import os
import csv
import tempfile
import numpy as np
import pytest

# Import the functions from confusion_matrix.py
from src.confusion_matrix import load_lines, compute_confusion_matrix
# Import the key classes and functions from problem.py
from src.poetry.trainer.problem import HumanPhraseData, Classifier, EOS

# -------------------------
# Tests for confusion_matrix.py
# -------------------------

def test_compute_confusion_matrix():
    # Create sample lists of input and target labels.
    input_lines = ["paraphrase", "not_paraphrase", "paraphrase", "not_paraphrase"]
    target_lines = ["1", "0", "0", "1"]
    tp, tn, fp, fn = compute_confusion_matrix(input_lines, target_lines)
    # For these inputs, expect:
    # - First pair: "paraphrase" with "1" => TP (+1)
    # - Second: "not_paraphrase" with "0" => TN (+1)
    # - Third: "paraphrase" with "0" => FN (+1)
    # - Fourth: "not_paraphrase" with "1" => FP (+1)
    assert tp == 1
    assert tn == 1
    assert fp == 1
    assert fn == 1

def test_load_lines(tmp_path):
    # Create a temporary file with three lines
    file = tmp_path / "temp.txt"
    file.write_text("line1\nline2\nline3")
    lines = load_lines(str(file))
    assert lines == ["line1", "line2", "line3"]

# -------------------------
# Fixtures for TSV data used by HumanPhraseData
# -------------------------

@pytest.fixture
def tsv_files(tmp_path):
    # Create temporary TSV files for training and evaluation.
    train_file = tmp_path / "parabank.tsv"
    eval_file = tmp_path / "eval.tsv"
    # Each row: first column is a human sentence (label 1), subsequent columns are machine-generated (label 0)
    train_file.write_text("Human sentence 1\tParaphrased sentence 1\nHuman sentence 2\tParaphrased sentence 2\n")
    eval_file.write_text("Human sentence 3\tParaphrased sentence 3\n")
    return str(train_file), str(eval_file)

# -------------------------
# Tests for HumanPhraseData functionality in problem.py
# -------------------------

def test_load_data(tsv_files):
    train_file, _ = tsv_files
    data_handler = HumanPhraseData(train_file, train_file, vocab_type="character")
    data = data_handler.load_data(train_file)
    # Expect the first row to yield two examples:
    #   ("Human sentence 1", 1) and ("Paraphrased sentence 1", 0)
    assert data[0] == ("Human sentence 1", 1)
    assert data[1] == ("Paraphrased sentence 1", 0)

def test_build_vocab_and_encode(tsv_files):
    train_file, _ = tsv_files
    data_handler = HumanPhraseData(train_file, train_file, vocab_type="character")
    data = data_handler.load_data(train_file)
    data_handler.build_vocab(data)
    vocab = data_handler.vocab
    # Check that the vocabulary is a dict and that it includes the special tokens.
    assert isinstance(vocab, dict)
    assert "<pad>" in vocab
    assert "<EOS>" in vocab
    # Test encode_data: for a sample text "ab", the encoded version should end with the EOS token.
    sample_text = "ab"
    encoded = data_handler.encode_data([(sample_text, 1)])
    tokens, label = encoded[0]
    # In character mode, the EOS token is appended at the end.
    assert tokens[-1] == EOS

def test_data_generator(tsv_files):
    train_file, _ = tsv_files
    data_handler = HumanPhraseData(train_file, train_file, vocab_type="character")
    data = data_handler.load_data(train_file)
    data_handler.build_vocab(data)
    encoded_data = data_handler.encode_data(data)
    # Create a data generator with a batch size of 2.
    gen = data_handler.data_generator(encoded_data, batch_size=2, shuffle=False)
    batch = next(gen)
    inputs, targets, weights = batch
    # Check that the inputs and targets are NumPy arrays and that batch size is correct.
    assert isinstance(inputs, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert inputs.shape[0] == 2
    assert targets.shape[0] == 2

def test_get_data_streams(tsv_files):
    train_file, eval_file = tsv_files
    data_handler = HumanPhraseData(train_file, eval_file, vocab_type="character")
    train_stream, eval_stream = data_handler.get_data_streams(batch_size=2)
    # Get one batch from each stream.
    train_batch = next(train_stream())
    eval_batch = next(eval_stream())
    for batch in (train_batch, eval_batch):
        inputs, targets, weights = batch
        assert isinstance(inputs, np.ndarray)
        assert isinstance(targets, np.ndarray)

def test_get_vocab_size(tsv_files):
    train_file, _ = tsv_files
    data_handler = HumanPhraseData(train_file, train_file, vocab_type="character")
    data = data_handler.load_data(train_file)
    data_handler.build_vocab(data)
    vocab_size = data_handler.get_vocab_size()
    # In character mode, vocab_size should equal the number of entries in the vocab dict and be at least 2.
    assert vocab_size == len(data_handler.vocab)
    assert vocab_size >= 2

# -------------------------
# Tests for the Classifier model
# -------------------------

def test_classifier_output_shape():
    # For testing, assume an arbitrary vocab size (e.g. 100).
    vocab_size = 100
    model = Classifier(vocab_size, mode='train')
    # Create a dummy batch of input sequences (batch_size x sequence_length)
    batch_size = 4
    seq_len = 10
    dummy_input = np.random.randint(0, vocab_size, (batch_size, seq_len))
    # Pass the dummy input through the model.
    output = model(dummy_input)
    # The model is built as: Embedding -> Mean (averaged over sequence) -> Dense(256) -> Relu -> Dense(2) -> LogSoftmax.
    # Thus, the output should have shape (batch_size, 2).
    assert output.shape[0] == batch_size
    assert output.shape[1] == 2
