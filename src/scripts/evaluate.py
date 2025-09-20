#!/usr/bin/env python3
"""
Evaluation Script for the Human/Machine Sentence Classifier using Google Gemini API or OpenRouter.

This script loads evaluation data, classifies each sentence using the selected provider,
and computes metrics including confusion matrix, accuracy, precision, recall, and F1-score.
"""

import argparse
import os
import json
import logging
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Support both execution contexts:
# - When run as a module/package with PYTHONPATH including 'src' (pytest.ini sets pythonpath = . src)
# - When imported directly by tests
try:
    from classifier.data import load_data_from_tsv
    from classifier.model import (
        classify_with_gemini,
        DEFAULT_GEMINI_MODEL,
        classify_with_openrouter,
        DEFAULT_OPENROUTER_MODEL,
        classify_text,
        ClassificationResult,
    )
    from classifier.config import get_config
    from classifier.cache import get_cache
except Exception:
    # Fallback for environments resolving relative imports within package context
    from ..classifier.data import load_data_from_tsv  # type: ignore
    from ..classifier.model import (  # type: ignore
        classify_with_gemini,
        DEFAULT_GEMINI_MODEL,
        classify_with_openrouter,
        DEFAULT_OPENROUTER_MODEL,
        classify_text,
        ClassificationResult,
    )
    from ..classifier.config import get_config  # type: ignore
    from ..classifier.cache import get_cache  # type: ignore

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
    """
    if labels is None or preds is None or len(labels) != len(preds):
        logging.error("Invalid input for compute_metrics. Labels or preds empty or lengths differ.")
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'support': 0
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0,1])

    # Guarantee 2x2 matrix
    if cm.shape != (2, 2):
        cm_fixed = np.zeros((2, 2), dtype=int)
        # Map existing dims into fixed if necessary
        try:
            cm_fixed[:cm.shape[0], :cm.shape[1]] = cm
            cm = cm_fixed
        except Exception:
            cm = np.array([[0, 0], [0, 0]])

    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': cm.tolist(),
        'support': int(len(labels))
    }

def main(args):
    """
    Main function to load data, run classification, evaluate, and save results.
    """
    logging.info("Starting evaluation process...")
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

    # --- 2. Run Classification with selected provider ---
    logging.info(f"Running classification using {args.provider} provider")
    true_labels = []
    predicted_labels = []
    api_errors = 0
    total_response_time = 0.0
    successful_classifications = 0
    start_time = time.time()

    # Use tqdm for progress bar if available
    data_iterator = tqdm(eval_data, desc="Classifying") if tqdm else eval_data

    for i, item in enumerate(data_iterator):
        text = item.get("text")
        true_label = item.get("label")

        if text is None or true_label is None:
            logging.warning(f"Skipping invalid item at index {i}: {item}")
            continue

        # Call provider API using unified function
        result = classify_text(
            text,
            provider=args.provider,
            gemini_model=args.gemini_model_name,
            openrouter_model=args.openrouter_model_name,
            max_retries=3
        )

        if result.is_success:
            try:
                predicted_label = int(result.label)
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
                successful_classifications += 1
                if result.response_time:
                    total_response_time += result.response_time
            except ValueError:
                logging.error(f"Invalid label format '{result.label}' for text: '{text[:50]}...'")
                api_errors += 1
        else:
            # Handle API error or None response
            logging.warning(f"Failed to get valid classification for item {i}: {result.error}")
            api_errors += 1

        # Optional: Add delay between API calls to avoid rate limits
        if args.delay > 0:
            time.sleep(args.delay)

        # Print progress periodically if tqdm is not available
        if not tqdm and (i + 1) % 50 == 0:
            avg_response_time = total_response_time / max(successful_classifications, 1)
            logging.info(f"Processed {i+1}/{len(eval_data)} items. "
                        f"Success rate: {successful_classifications/(i+1)*100:.1f}%. "
                        f"Avg response time: {avg_response_time:.2f}s")


    end_time = time.time()
    duration = end_time - start_time

    # Calculate performance metrics
    avg_response_time = total_response_time / max(successful_classifications, 1)
    success_rate = successful_classifications / len(eval_data) * 100

    logging.info(f"Classification finished in {duration:.2f} seconds.")
    logging.info(f"Total items processed: {len(eval_data)}")
    logging.info(f"Successful classifications: {successful_classifications}")
    logging.info(f"API errors/invalid responses: {api_errors}")
    logging.info(f"Success rate: {success_rate:.1f}%")
    logging.info(f"Average response time: {avg_response_time:.2f}s")

    if not predicted_labels:
        logging.error("No successful classifications were made. Cannot compute metrics.")
        return

    # --- 3. Compute Metrics ---
    logging.info("Computing evaluation metrics...")
    eval_results = compute_metrics(true_labels, predicted_labels)

    # Add run information to results
    eval_results['provider'] = args.provider
    eval_results['model_used'] = args.gemini_model_name if args.provider == "gemini" else args.openrouter_model_name
    eval_results['eval_file'] = args.eval_file
    eval_results['total_samples'] = len(eval_data)
    eval_results['successful_classifications'] = successful_classifications
    eval_results['api_errors'] = api_errors
    eval_results['duration_seconds'] = duration
    eval_results['average_response_time'] = avg_response_time
    eval_results['success_rate'] = success_rate

    # Add cache statistics if available
    if get_cache:
        cache = get_cache()
        cache_stats = cache.get_stats()
        eval_results['cache_stats'] = cache_stats

    # --- 4. Print and Save Results ---
    print(f"\n--- {args.provider.capitalize()} Evaluation Results ---")
    print("=" * 50)

    # Performance metrics
    print("PERFORMANCE METRICS:")
    print(f"  Provider: {eval_results['provider']}")
    print(f"  Model: {eval_results['model_used']}")
    print(f"  Total samples: {eval_results['total_samples']}")
    print(f"  Successful classifications: {eval_results['successful_classifications']}")
    print(f"  API errors: {eval_results['api_errors']}")
    print(f"  Success rate: {eval_results['success_rate']:.1f}%")
    print(f"  Average response time: {eval_results['average_response_time']:.2f}s")
    print(f"  Total duration: {eval_results['duration_seconds']:.2f}s")

    print("\nCLASSIFICATION METRICS:")
    # Classification metrics
    for key, value in eval_results.items():
        if key in ['provider', 'model_used', 'eval_file', 'total_samples',
                   'successful_classifications', 'api_errors', 'duration_seconds',
                   'average_response_time', 'success_rate']:
            continue  # Skip performance metrics already shown
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
    # Keep filename stable for backward compatibility, but include provider in the JSON content
    output_eval_file = os.path.join(args.output_dir, "gemini_evaluation_results.json")
    logging.info(f"\nSaving evaluation results to: {output_eval_file}")
    try:
        with open(output_eval_file, "w", encoding='utf-8') as writer:
            json.dump(eval_results, writer, indent=4)
    except Exception as e:
        logging.error(f"Failed to save results to {output_eval_file}: {e}")

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    # Get configuration for defaults
    config = get_config() if get_config else None

    parser = argparse.ArgumentParser(
        description="Evaluate sentence classification using Google Gemini API or OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to the evaluation data TSV file (text<tab>label)."
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "openrouter"],
        default=config.default_provider if config else "openrouter",
        help="Provider to use for classification."
    )
    parser.add_argument(
        "--gemini_model_name",
        type=str,
        default=config.effective_gemini_model if config else DEFAULT_GEMINI_MODEL,
        help="Name of the Gemini model to use."
    )
    parser.add_argument(
        "--openrouter_model_name",
        type=str,
        default=config.effective_openrouter_model if config else DEFAULT_OPENROUTER_MODEL,
        help="Name of the OpenRouter model to use."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.output_dir if config else "results",
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=config.evaluation_delay if config else 0.1,
        help="Optional delay (seconds) between API calls to avoid rate limits."
    )
    # Removed args: --model_dir, --max_length, --per_device_eval_batch_size, --force_cpu

    args = parser.parse_args()
    main(args)