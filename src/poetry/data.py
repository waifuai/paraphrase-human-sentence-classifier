#!/usr/bin/env python3
"""
Data handling module for the Paraphrase Classifier.
Contains the HumanPhraseData class for loading, preprocessing, and generating data batches.
"""

import csv
import random as rnd
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
import trax

# Constant for end-of-sentence token
EOS = 1


class HumanPhraseData:
    """
    Handles data loading, preprocessing, and vocabulary building for the classifier.
    """

    def __init__(self, train_file: str, eval_file: str, vocab_type: str = "subword", vocab_size: int = 2**14):
        self.train_file = train_file
        self.eval_file = eval_file
        self.vocab_type = vocab_type
        self.vocab_size = vocab_size
        self.vocab: Any = None

    def load_data(self, filename: str) -> List[Tuple[str, int]]:
        """
        Loads data from a TSV file.
        The first column is treated as the positive (human-generated) example.
        Subsequent columns are treated as negative (machine-generated) examples.
        """
        data = []
        # Ensure filename is treated as relative to the project root or an absolute path
        filepath = Path(filename)
        if not filepath.exists():
             # Attempt relative path from src if not found directly
             filepath = Path('..') / filename # Assuming script runs from src/ or similar
             if not filepath.exists():
                 raise FileNotFoundError(f"Data file not found: {filename}")

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) > 1:
                    for i, phrase in enumerate(row):
                        label = 1 if i == 0 else 0  # First column positive; others negative
                        data.append((phrase, label))
        return data

    def build_vocab(self, data: List[Tuple[str, int]]) -> None:
        """
        Builds a vocabulary from the data.
        For subword tokenization, uses Trax's Tokenize.
        For character tokenization, builds a character-to-index mapping.
        """
        if self.vocab_type == "subword":
            all_text = [text for text, _ in data]
            # Using Trax's Tokenize to build a subword vocabulary.
            # Ensure vocab_dir is specified if needed, or handle potential default locations.
            self.vocab = trax.data.Tokenize(vocab_file=None, vocab_dir=None, vocab_size=self.vocab_size, keys=("text",))
            # Initialize the tokenizer with the provided data (using a dummy filename of None)
            self.vocab.init_from_file(None, data=all_text)
        elif self.vocab_type == "character":
            unique_chars = set()
            for text, _ in data:
                unique_chars.update(list(text))
            # Add special tokens: <pad> and <EOS>
            special_tokens = ['<pad>', '<EOS>']
            vocab_list = special_tokens + sorted(list(unique_chars))
            self.vocab = {char: idx for idx, char in enumerate(vocab_list)}
        else:
            raise ValueError(f"Unsupported vocab type: {self.vocab_type}")

    def encode_data(self, data: List[Tuple[str, int]]) -> List[Tuple[List[int], int]]:
        """
        Encodes the text data using the built vocabulary.
        For subword, applies the tokenizer.
        For character, maps each character to its index and appends an EOS token.
        """
        if self.vocab is None:
             raise RuntimeError("Vocabulary has not been built yet. Call build_vocab first.")

        encoded_data = []
        for text, label in data:
            if self.vocab_type == "subword":
                # The tokenizer returns a list of token IDs when called on text.
                encoded_text = self.vocab(text)
            elif self.vocab_type == "character":
                encoded_text = [self.vocab.get(char, self.vocab['<pad>']) for char in text] + [EOS] # Use get for safety
            else:
                raise ValueError(f"Unsupported vocab type: {self.vocab_type}")
            encoded_data.append((list(encoded_text), label))
        return encoded_data

    def data_generator(self, data: List[Tuple[List[int], int]], batch_size: int, shuffle: bool = True) -> Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yields batches of encoded data.
        Each batch is padded so that all sequences share the same length.
        """
        if not data:
             return lambda: iter([]) # Return empty generator if no data

        epoch_data = list(data) # Create a copy for shuffling per epoch
        if shuffle:
            rnd.shuffle(epoch_data)

        idx = 0
        while idx < len(epoch_data):
            batch = epoch_data[idx : idx + batch_size]
            idx += batch_size

            if not batch: continue # Should not happen with check above, but safety

            max_len = max(len(seq) for seq, _ in batch)
            # Pad sequences with 0 (assuming 0 is the padding token index)
            inputs = np.array([seq + [0] * (max_len - len(seq)) for seq, _ in batch], dtype=np.int32)
            targets = np.array([label for _, label in batch], dtype=np.int32)
            # Weights: typically 1 for real tokens, 0 for padding. Here, just 1s for sequence classification.
            weights = np.ones_like(targets, dtype=np.float32)

            yield (inputs, targets, weights)


    def get_data_streams(self, batch_size: int) -> Tuple[Callable, Callable]:
        """
        Loads data, builds vocab, encodes data, and returns lambda functions
        for the training and evaluation data generators.
        """
        # Load data
        train_data_raw = self.load_data(self.train_file)
        eval_data_raw = self.load_data(self.eval_file)

        # Build vocabulary only from training data
        self.build_vocab(train_data_raw)

        # Encode data
        encoded_train_data = self.encode_data(train_data_raw)
        encoded_eval_data = self.encode_data(eval_data_raw)

        # Create generator functions
        # Need to wrap the generator call in a lambda to make it restartable for each epoch
        train_stream = lambda: self.data_generator(encoded_train_data, batch_size, shuffle=True)
        eval_stream = lambda: self.data_generator(encoded_eval_data, batch_size, shuffle=False)

        return train_stream, eval_stream

    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.
        """
        if self.vocab is None:
             raise RuntimeError("Vocabulary has not been built yet.")

        if self.vocab_type == "subword":
            # For trax.data.Tokenize, vocab_size is usually pre-defined or accessible via an attribute
            # Let's rely on the initialized vocab_size attribute for consistency
            return self.vocab_size
        elif self.vocab_type == "character":
            return len(self.vocab)
        else:
            raise ValueError(f"Unsupported vocab type: {self.vocab_type}")