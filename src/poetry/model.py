#!/usr/bin/env python3
"""
Model definition for the Paraphrase Classifier.
"""

from trax import layers as tl

def Classifier(vocab_size: int, d_model: int = 512, d_ff: int = 256, n_classes: int = 2, mode: str = 'train') -> tl.Serial:
    """
    Defines a simple Transformer-based classifier model.

    Args:
        vocab_size: Size of the input vocabulary.
        d_model: Dimension of embedding and model layers.
        d_ff: Dimension of the feed-forward layer.
        n_classes: Number of output classes.
        mode: 'train', 'eval', or 'predict'.

    Returns:
        A Trax Serial model.
    """
    # TODO: Consider making the architecture more configurable (e.g., number of layers, heads if using Transformer blocks)
    return tl.Serial(
        tl.Embedding(vocab_size, d_model),
        tl.Mean(axis=1),  # Average embeddings across the sequence length dimension
        tl.Dense(d_ff),
        tl.Relu(),
        tl.Dense(n_classes),
        tl.LogSoftmax()  # Use LogSoftmax for numerical stability with CrossEntropyLoss
    )