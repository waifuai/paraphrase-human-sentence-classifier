#!/usr/bin/env python3
"""
Evaluation Script for the Human/Machine Sentence Classifier using Google Gemini API.

This script loads evaluation data, classifies each sentence using the Gemini API,
and computes metrics including confusion matrix, accuracy, precision, recall, and F1-score.
"""

import argparse
import os
import json
import logging
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Use relative imports assuming 'src' is in PYTHONPATH or running as module
from ..classifier.data import load_data_from_tsv
from ..classifier.model import classify_with_gemini, DEFAULT_GEMINI_MODEL

# Attempt to import tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # Set to None if not installed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics(labels, preds):
    """
    Computes evaluation metrics from predictions.
    Modified to take lists of labels and predictions directly.

    Args:
        labels: A list of true integer labels (0 or 1).
        preds: A list of predicted integer labels (0 or 1).

    Returns:
        A dictionary containing accuracy, precision, recall, f1, and confusion matrix.
    """
    if not labels or not preds or len(labels) != len(preds):
         logging.error("Invalid input for compute_metrics. Labels or preds empty or lengths differ.")
         return {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'confusion_matrix': [[0,0],[0,0]], 'support': 0}

    precision, recall, f1, support_tuple = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    support = len(labels) # Total number of samples evaluated

    # Ensure confusion matrix is 2x2, padding if necessary
    if cm.shape == (1, 1):
        # Determine which label was present based on the first label
        present_label = labels[0]
        if present_label == 0: # Only label 0 present/predicted
             cm_full = np.array([[cm[0,0], 0], [0, 0]])
        else: # Only label 1 present/predicted
             cm_full = np.array([[0, 0], [0, cm[0,0]]])
        cm = cm_full
    elif cm.shape != (2, 2):
         # Handle unexpected shapes, default to zero matrix
         logging.warning(f"Unexpected confusion matrix shape {cm.shape}. Defaulting.")
         cm = np.array([[0, 0], [0, 0]])

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist(), # Convert numpy array to list for JSON
        'support': support # Add total support count
    }

def main(args):
    """
    Main function to load data, run Gemini classification, evaluate, and save results.
    """
    logging.info("Starting evaluation process with Gemini API...")
    logging.info(f"Configuration: {args}")

    # --- 1. Load Evaluation Data ---
    logging.info(f"Loading evaluation data from: {args.eval_file}")
    try:
        datasets = load_data_from_tsv(train_file=None, eval_file=args.eval_file)
        eval_data = datasets.get('test', [])
        if not eval_data:
            logging.error(f"No evaluation data loaded from {args.eval_file}. Exiting.")
            return
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    logging.info(f"Evaluation data size: {len(eval_data)}")

    # --- 2. Run Classification with Gemini ---
    logging.info(f"Running classification using Gemini model: {args.gemini_model_name}")
    true_labels = []
    predicted_labels = []
    api_errors = 0
    start_time = time.time()

    # Use tqdm for progress bar if available
    data_iterator = tqdm(eval_data, desc="Classifying") if tqdm else eval_data

    for i, item in enumerate(data_iterator):
        text = item.get("text")
        true_label = item.get("label")

        if text is None or true_label is None:
            logging.warning(f"Skipping invalid item at index {i}: {item}")
            continue

        # Call Gemini API
        predicted_label_str = classify_with_gemini(text, model_name=args.gemini_model_name)

        if predicted_label_str is not None:
            try:
                predicted_label = int(predicted_label_str)
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
            except ValueError:
                 logging.error(f"Gemini returned non-integer value '{predicted_label_str}' for text: '{text[:50]}...'")
                 api_errors += 1
        else:
            # Handle API error or None response (e.g., unexpected format)
            logging.warning(f"Failed to get valid classification for item {i}. Text: '{text[:50]}...'")
            api_errors += 1
            # Optional: Add a placeholder prediction (e.g., opposite of true label)
            # or simply skip this sample for metric calculation. Skipping for now.

        # Optional: Add delay between API calls to avoid rate limits
        if args.delay > 0:
            time.sleep(args.delay)

        # Print progress periodically if tqdm is not available
        if not tqdm and (i + 1) % 50 == 0:
             logging.info(f"Processed {i+1}/{len(eval_data)} items...")


    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Classification finished in {duration:.2f} seconds.")
    logging.info(f"Total items processed: {len(eval_data)}")
    logging.info(f"Successful classifications: {len(predicted_labels)}")
    logging.info(f"API errors/invalid responses: {api_errors}")

    if not predicted_labels:
        logging.error("No successful classifications were made. Cannot compute metrics.")
        return

    # --- 3. Compute Metrics ---
    logging.info("Computing evaluation metrics...")
    eval_results = compute_metrics(true_labels, predicted_labels)

    # Add run information to results
    eval_results['model_used'] = args.gemini_model_name
    eval_results['eval_file'] = args.eval_file
    eval_results['total_samples'] = len(eval_data)
    eval_results['successful_classifications'] = len(predicted_labels)
    eval_results['api_errors'] = api_errors
    eval_results['duration_seconds'] = duration

    # --- 4. Print and Save Results ---
    print("\n--- Gemini Evaluation Results ---")
    # Pretty print the results
    for key, value in eval_results.items():
        if key == "confusion_matrix":
            print(f"  Confusion Matrix (TN, FP / FN, TP):")
            print(f"    {value[0]}")
            print(f"    {value[1]}")
        elif isinstance(value, float):
            print(f"  {key.replace('_', ' ').capitalize()}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').capitalize()}: {value}")

    # Ensure results directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_eval_file = os.path.join(args.output_dir, "gemini_evaluation_results.json")
    logging.info(f"\nSaving evaluation results to: {output_eval_file}")
    try:
        with open(output_eval_file, "w", encoding='utf-8') as writer:
            json.dump(eval_results, writer, indent=4)
    except Exception as e:
        logging.error(f"Failed to save results to {output_eval_file}: {e}")

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sentence classification using Google Gemini API.")

    parser.add_argument("--eval_file", type=str, required=True, help="Path to the evaluation data TSV file (text<tab>label).")
    parser.add_argument("--gemini_model_name", type=str, default=DEFAULT_GEMINI_MODEL, help="Name of the Gemini model to use.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation results.")
    parser.add_argument("--delay", type=float, default=0.1, help="Optional delay (seconds) between API calls to avoid rate limits.")
    # Removed args: --model_dir, --max_length, --per_device_eval_batch_size, --force_cpu

    args = parser.parse_args()
    main(args)