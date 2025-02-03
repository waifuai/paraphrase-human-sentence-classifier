#!/usr/bin/env python3
"""
Training script for the Paraphrase - Human Sentence Classifier using Trax.
It loads data, builds a vocabulary, defines a Transformer-based classifier, and trains the model.
"""

import csv
import os
import random as rnd
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
import trax
from trax import layers as tl
from trax.supervised import training

# Initialize CPU environment (modify for GPU/TPU if needed)
trax.supervised.trainer_lib.init_cpu_environment()

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
        with open(filename, 'r', encoding='utf-8') as f:
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
        encoded_data = []
        for text, label in data:
            if self.vocab_type == "subword":
                # The tokenizer returns a list of token IDs when called on text.
                encoded_text = self.vocab(text)
            elif self.vocab_type == "character":
                encoded_text = [self.vocab[char] for char in text] + [EOS]
            else:
                raise ValueError(f"Unsupported vocab type: {self.vocab_type}")
            encoded_data.append((list(encoded_text), label))
        return encoded_data

    def data_generator(self, data: List[Tuple[List[int], int]], batch_size: int, shuffle: bool = True) -> Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yields batches of encoded data.
        Each batch is padded so that all sequences share the same length.
        """
        if shuffle:
            rnd.shuffle(data)

        batch = []
        for example in data:
            batch.append(example)
            if len(batch) == batch_size:
                max_len = max(len(seq) for seq, _ in batch)
                inputs = np.array([seq + [0] * (max_len - len(seq)) for seq, _ in batch])
                targets = np.array([label for _, label in batch])
                yield (inputs, targets, np.ones_like(targets))
                batch = []

        if batch:
            max_len = max(len(seq) for seq, _ in batch)
            inputs = np.array([seq + [0] * (max_len - len(seq)) for seq, _ in batch])
            targets = np.array([label for _, label in batch])
            yield (inputs, targets, np.ones_like(targets))

    def get_data_streams(self, batch_size: int) -> Tuple[Callable, Callable]:
        """
        Returns lambda functions for the training and evaluation data generators.
        """
        train_data = self.load_data(self.train_file)
        eval_data = self.load_data(self.eval_file)

        # Build vocabulary only from training data
        self.build_vocab(train_data)

        encoded_train_data = self.encode_data(train_data)
        encoded_eval_data = self.encode_data(eval_data)

        train_stream = lambda: self.data_generator(encoded_train_data, batch_size, shuffle=True)
        eval_stream = lambda: self.data_generator(encoded_eval_data, batch_size, shuffle=False)

        return train_stream, eval_stream

    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.
        """
        if self.vocab_type == "subword":
            # Assuming a fixed vocab_size for subword tokenization
            return self.vocab_size
        elif self.vocab_type == "character":
            return len(self.vocab)
        else:
            raise ValueError(f"Unsupported vocab type: {self.vocab_type}")


def Classifier(vocab_size: int, mode: str = 'train') -> tl.Serial:
    """
    Defines a simple Transformer-based classifier model.
    """
    return tl.Serial(
        tl.Embedding(vocab_size, 512),
        tl.Mean(axis=1),  # Average over the sequence dimension
        tl.Dense(256),
        tl.Relu(),
        tl.Dense(2),
        tl.LogSoftmax()  # Output probabilities for two classes
    )


def train_model(model: tl.Serial, train_stream: Callable, eval_stream: Callable, output_dir: str, n_steps: int = 1000) -> training.Loop:
    """
    Trains the model using the provided training and evaluation data streams.
    """
    train_task = training.TrainTask(
        labeled_data=train_stream(),
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=100,
    )

    eval_task = training.EvalTask(
        labeled_data=eval_stream(),
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
    )

    training_loop = training.Loop(model, train_task, eval_tasks=[eval_task], output_dir=output_dir)
    training_loop.run(n_steps=n_steps)
    return training_loop


def main():
    # File paths (update as needed)
    train_file = "parabank.tsv"
    eval_file = "eval.tsv"
    output_dir = "model_output"

    # Training parameters
    batch_size = 16
    n_steps = 500  # Adjust as needed for demonstration or full training

    # Choose vocabulary type: 'subword' or 'character'
    vocab_type = "subword"

    # Data handling
    data_handler = HumanPhraseData(train_file, eval_file, vocab_type=vocab_type)
    train_stream, eval_stream = data_handler.get_data_streams(batch_size)

    # Determine vocabulary size based on tokenization type
    vocab_size = data_handler.get_vocab_size()

    # Model initialization
    model = Classifier(vocab_size)

    # Train the model
    train_model(model, train_stream, eval_stream, output_dir, n_steps)
    print("Training complete. Model saved in:", output_dir)


if __name__ == "__main__":
    main()
