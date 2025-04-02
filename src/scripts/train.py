#!/usr/bin/env python3
"""
Main training script for the Human/Paraphrase Sentence Classifier using Hugging Face Transformers.

This script orchestrates the data loading, model definition, training,
and saving of the classifier model using the Trainer API.
"""

import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Ensure imports work correctly when running as a module (e.g., python -m src.scripts.train)
# Use relative imports within the 'src' package
from ..classifier.data import load_and_tokenize_data
from ..classifier.model import load_model

# Import necessary Hugging Face classes
from transformers import Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
import torch # Import torch to check for GPU availability

def compute_metrics(pred):
    """
    Computes evaluation metrics from predictions.

    Args:
        pred: An EvalPrediction object containing predictions and label_ids.

    Returns:
        A dictionary containing accuracy, precision, recall, f1, and confusion matrix.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds).tolist() # Convert numpy array to list for JSON serialization

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm # Include confusion matrix in results
    }

def main(args):
    """
    Main function to set up and run the training process.
    """
    print("Starting training process with Hugging Face Trainer...")
    print(f"Configuration: {args}")

    # --- 1. Load Tokenizer and Datasets ---
    print("Loading and tokenizing data...")
    # Tokenizer is loaded inside load_and_tokenize_data now
    tokenized_datasets = load_and_tokenize_data(
        train_file=args.train_file,
        eval_file=args.eval_file,
        tokenizer_name=args.model_name_or_path, # Use same model name for tokenizer
        max_length=args.max_length,
        test_size=args.test_size # Used if eval_file is None
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"] # Use 'test' split as eval

    print(f"Training data size: {len(train_dataset)}")
    print(f"Evaluation data size: {len(eval_dataset)}")

    # --- 2. Load Model ---
    print("Loading pre-trained model...")
    model = load_model(
        model_name=args.model_name_or_path,
        num_labels=args.num_labels
    )

    # --- 3. Define Training Arguments ---
    print("Defining training arguments...")
    # Check for GPU availability
    use_gpu = torch.cuda.is_available() and not args.force_cpu
    print(f"Using GPU: {use_gpu}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="no",          # Do not save checkpoints during training
        logging_strategy="epoch",    # Log metrics at the end of each epoch
        load_best_model_at_end=False,# Cannot load best model if not saving checkpoints
        metric_for_best_model="f1",  # Use F1 score to determine the best model
        greater_is_better=True,      # Higher F1 is better
        report_to="none",            # Disable external reporting (like wandb) unless configured
        fp16=use_gpu,                # Enable mixed precision if GPU is available
        # no_cuda=not use_gpu,         # Explicitly disable CUDA if force_cpu is True or no GPU
        push_to_hub=False,           # Do not push to Hugging Face Hub by default
        # Add other arguments as needed
    )

    # --- 4. Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path), # Pass tokenizer for padding consistency
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] # Add early stopping
    )

    # --- 5. Train ---
    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    # --- 6. Save Model, Tokenizer & Metrics ---
    print(f"Saving best model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    # Tokenizer is usually saved automatically with save_model, but explicit save is safe
    # tokenizer.save_pretrained(args.output_dir) # Already done by Trainer

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # Saves optimizer state, scheduler, etc.

    # --- 7. Evaluate on Test Set ---
    print("Evaluating final model on the evaluation set...")
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Evaluation Metrics: {eval_metrics}")
    # trainer.log_metrics("eval", eval_metrics) # Commented out: Causes formatting error with list (confusion matrix)
    trainer.save_metrics("eval", eval_metrics)

    print(f"Training complete. Best model and metrics saved in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hugging Face Transformer model for sentence classification.")

    # Data arguments
    parser.add_argument("--train_file", type=str, default="data/train.tsv", help="Path to the training data TSV file.")
    parser.add_argument("--eval_file", type=str, default="data/eval.tsv", help="Path to the evaluation data TSV file. If None, train_file is split.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of training data for evaluation if eval_file is None.")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased", help="Name/path of the pre-trained model.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classification labels.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")

    # Training arguments (matching TrainingArguments)
    parser.add_argument("--output_dir", type=str, default="model_output_hf", help="Directory to save model checkpoints and outputs.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Total number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size per device during evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping.")
    parser.add_argument("--force_cpu", action="store_true", help="Force training on CPU even if GPU is available.")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)