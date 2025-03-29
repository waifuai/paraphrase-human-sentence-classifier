#!/usr/bin/env python3
"""
Main training script for the Paraphrase - Human Sentence Classifier.

This script orchestrates the data loading, model definition, training,
and saving of the classifier model using the Trax library.
"""

import argparse
import os
from typing import Callable

import trax
from trax import layers as tl
from trax.supervised import training

# Assuming src is in PYTHONPATH (as per pytest.ini)
from poetry.data import HumanPhraseData
from poetry.model import Classifier

# Initialize CPU environment (modify for GPU/TPU if needed)
# Consider making device selection configurable
trax.supervised.trainer_lib.init_cpu_environment()


def train_model(model: tl.Serial, train_stream: Callable, eval_stream: Callable, output_dir: str, n_steps: int = 1000, checkpoint_steps: int = 100) -> training.Loop:
    """
    Trains the model using the provided training and evaluation data streams.

    Args:
        model: The Trax model to train.
        train_stream: A callable that yields training batches.
        eval_stream: A callable that yields evaluation batches.
        output_dir: Directory to save checkpoints and model artifacts.
        n_steps: Total number of training steps.
        checkpoint_steps: Frequency (in steps) for saving checkpoints.

    Returns:
        The completed Trax training loop object.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    train_task = training.TrainTask(
        labeled_data=train_stream(),
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.01), # TODO: Make learning rate configurable
        n_steps_per_checkpoint=checkpoint_steps,
    )

    eval_task = training.EvalTask(
        labeled_data=eval_stream(),
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
    )

    training_loop = training.Loop(model, train_task, eval_tasks=[eval_task], output_dir=output_dir)
    training_loop.run(n_steps=n_steps)
    return training_loop


def main(args):
    """
    Main function to set up and run the training process.
    """
    print("Starting training process...")
    print(f"Configuration: {args}")

    # Data handling
    # Note: File paths in HumanPhraseData might need adjustment if they are relative
    # and this script is run from a different directory than the original.
    # Consider passing absolute paths or paths relative to a defined root dir.
    data_handler = HumanPhraseData(
        train_file=args.train_file,
        eval_file=args.eval_file,
        vocab_type=args.vocab_type,
        vocab_size=args.vocab_size
    )
    train_stream, eval_stream = data_handler.get_data_streams(args.batch_size)

    # Determine vocabulary size based on tokenization type
    vocab_size = data_handler.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # Model initialization
    # TODO: Make model dimensions configurable
    model = Classifier(vocab_size=vocab_size, mode='train')
    print("Model initialized.")
    # print(model) # Optional: print model structure

    # Train the model
    print(f"Starting training for {args.n_steps} steps...")
    train_model(
        model,
        train_stream,
        eval_stream,
        args.output_dir,
        args.n_steps,
        args.checkpoint_steps
    )
    print(f"Training complete. Model saved in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Paraphrase Classifier model.")

    # Data arguments
    parser.add_argument("--train_file", type=str, default="parabank.tsv", help="Path to the training data TSV file.")
    parser.add_argument("--eval_file", type=str, default="eval.tsv", help="Path to the evaluation data TSV file.")
    parser.add_argument("--output_dir", type=str, default="model_output", help="Directory to save model checkpoints and outputs.")

    # Vocabulary arguments
    parser.add_argument("--vocab_type", type=str, default="subword", choices=["subword", "character"], help="Type of vocabulary to use.")
    parser.add_argument("--vocab_size", type=int, default=2**14, help="Vocabulary size (used for subword tokenization).")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--n_steps", type=int, default=500, help="Total number of training steps.")
    parser.add_argument("--checkpoint_steps", type=int, default=100, help="Save checkpoint every N steps.")
    # TODO: Add arguments for learning rate, model dimensions, device selection etc.

    args = parser.parse_args()
    main(args)