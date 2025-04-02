#!/usr/bin/env python3
"""
Evaluation Script for the Human/Paraphrase Sentence Classifier using Hugging Face Transformers.

This script loads a trained model and tokenizer, runs predictions on an evaluation dataset,
and computes various metrics including confusion matrix, accuracy, precision, recall, and F1-score.
"""

import argparse
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Ensure imports work correctly assuming 'src' is in PYTHONPATH
from classifier.data import load_and_tokenize_data # Use the updated data loader

# Import necessary Hugging Face classes
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch # Import torch to check for GPU availability

def compute_metrics(pred):
    """
    Computes evaluation metrics from predictions.
    (Same function as in train.py for consistency)

    Args:
        pred: An EvalPrediction object containing predictions and label_ids.

    Returns:
        A dictionary containing accuracy, precision, recall, f1, and confusion matrix.
    """
    labels = pred.label_ids
    # Ensure predictions are valid before argmax
    if pred.predictions is None or len(pred.predictions) == 0:
        print("Warning: No predictions found in EvalPrediction object.")
        return {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'confusion_matrix': [[0,0],[0,0]]}

    preds = np.argmax(pred.predictions, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    # Ensure confusion matrix is 2x2, padding if necessary (e.g., if only one class predicted)
    if cm.shape == (1, 1):
        if labels[0] == 0: # Only predicted/actual 0
             cm_full = np.array([[cm[0,0], 0], [0, 0]])
        else: # Only predicted/actual 1
             cm_full = np.array([[0, 0], [0, cm[0,0]]])
        cm = cm_full
    elif cm.shape != (2, 2):
         # Handle unexpected shapes if necessary, default to zero matrix
         print(f"Warning: Unexpected confusion matrix shape {cm.shape}. Defaulting.")
         cm = np.array([[0, 0], [0, 0]])


    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist() # Convert numpy array to list for JSON serialization
    }

def main(args):
    """
    Main function to load model, run evaluation, and print metrics.
    """
    print("Starting evaluation process...")
    print(f"Configuration: {args}")

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    # --- 1. Load Tokenizer and Model ---
    print(f"Loading tokenizer and model from: {args.model_dir}")
    # Use the same tokenizer name/path as the model was trained with
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # --- 2. Load and Tokenize Evaluation Data ---
    print(f"Loading and tokenizing evaluation data from: {args.eval_file}")
    # We only need the 'test' split for evaluation
    # Pass None for train_file as we are only evaluating
    tokenized_datasets = load_and_tokenize_data(
        train_file=None, # No training file needed for evaluation
        eval_file=args.eval_file,
        tokenizer_name=args.model_dir, # Use loaded tokenizer's path/name
        max_length=args.max_length
    )

    if "test" not in tokenized_datasets:
        raise ValueError(f"Could not load 'test' split from eval_file: {args.eval_file}")

    eval_dataset = tokenized_datasets["test"]
    print(f"Evaluation data size: {len(eval_dataset)}")

    # --- 3. Set up Trainer for Evaluation ---
    # We use Trainer just for its convenient evaluate method
    print("Setting up Trainer for evaluation...")
    # Check for GPU availability
    use_gpu = torch.cuda.is_available() and not args.force_cpu
    print(f"Using GPU for evaluation: {use_gpu}")

    # Minimal TrainingArguments needed for evaluation
    training_args = TrainingArguments(
        output_dir=args.model_dir, # Output dir isn't really used here, but required
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",
        fp16=use_gpu,
        # no_cuda=not use_gpu,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, # Pass tokenizer
    )

    # --- 4. Run Evaluation ---
    print("Running evaluation...")
    eval_results = trainer.evaluate()

    # --- 5. Print and Save Results ---
    print("\n--- Evaluation Results ---")
    # Pretty print the results
    for key, value in eval_results.items():
        if key == "eval_confusion_matrix":
            print(f"  Confusion Matrix:")
            print(f"    TN: {value[0][0]}, FP: {value[0][1]}")
            print(f"    FN: {value[1][0]}, TP: {value[1][1]}")
        else:
            # Format floats nicely
            if isinstance(value, float):
                print(f"  {key.replace('eval_', '').capitalize()}: {value:.4f}")
            else:
                print(f"  {key.replace('eval_', '').capitalize()}: {value}")

    # Optionally save results to a file
    output_eval_file = os.path.join(args.model_dir, "evaluation_results.json")
    print(f"\nSaving evaluation results to: {output_eval_file}")
    with open(output_eval_file, "w") as writer:
        json.dump(eval_results, writer, indent=4)

    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Hugging Face Transformer model for sentence classification.")

    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the saved model and tokenizer.")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to the evaluation data TSV file.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization (should match training).")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size per device during evaluation.")
    parser.add_argument("--force_cpu", action="store_true", help="Force evaluation on CPU even if GPU is available.")

    args = parser.parse_args()
    main(args)