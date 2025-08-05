#!/usr/bin/env python3
"""
Data handling module for the Human/Machine Sentence Classifier.
Loads data from TSV files into lists of dictionaries.
"""

import csv
import logging
import random
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_tsv(file_path: str) -> List[Dict[str, any]]:
    """Loads a TSV file into a list of dictionaries."""
    data: List[Dict[str, any]] = []
    try:
        with open(file_path, mode='r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            for i, row in enumerate(reader):
                if len(row) == 2:
                    text, label_str = row
                    try:
                        label = int(label_str)
                        if label not in (0, 1):
                            logging.warning(f"Invalid label '{label_str}' in {file_path} at line {i+1}. Skipping row.")
                            continue
                        data.append({"text": text, "label": label})
                    except ValueError:
                        logging.warning(f"Invalid label format '{label_str}' in {file_path} at line {i+1}. Skipping row.")
                else:
                    logging.warning(f"Skipping malformed row (expected 2 columns, got {len(row)}) in {file_path} at line {i+1}: {row}")
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        raise
    return data

def load_data_from_tsv(
    train_file: Optional[str],
    eval_file: Optional[str],
    test_size: float = 0.1,
    random_seed: int = 42 # Added for reproducible splits
) -> Dict[str, List[Dict[str, any]]]:
    """
    Loads data from TSV files and returns a dictionary of data splits.

    Args:
        train_file: Path to the training TSV file (columns: text, label). Can be None if only eval is needed.
        eval_file: Path to the evaluation TSV file (columns: text, label).
                   If None and train_file is provided, the train_file will be split.
        test_size: Fraction of training data to use for evaluation if eval_file is None.
        random_seed: Seed for random shuffling during split.

    Returns:
        A dictionary containing 'train' and 'test' splits as lists of dictionaries.
        Example: {'train': [{'text': '...', 'label': 1}, ...], 'test': [...]}
    """
    datasets = {}

    if train_file:
        logging.info(f"Loading training data from: {train_file}")
        train_data = _load_tsv(train_file)
        if eval_file:
            logging.info(f"Loading evaluation data from: {eval_file}")
            eval_data = _load_tsv(eval_file)
            datasets['train'] = train_data
            datasets['test'] = eval_data
        else:
            # Split train_data if no eval_file is provided
            if not train_data:
                raise FileNotFoundError("Cannot split data: Training data is empty or failed to load.")
            logging.info(f"No evaluation file provided. Splitting train data with test_size={test_size}")
            if not (0 < test_size < 1):
                raise ValueError("test_size must be between 0 and 1")

            random.seed(random_seed)
            random.shuffle(train_data)

            split_idx = int(len(train_data) * (1 - test_size))
            datasets['train'] = train_data[:split_idx]
            datasets['test'] = train_data[split_idx:]
            logging.info(f"Split complete. Train size: {len(datasets['train'])}, Test size: {len(datasets['test'])}")

    elif eval_file:
        # Only evaluation file provided
        logging.info(f"Loading evaluation data only from: {eval_file}")
        eval_data = _load_tsv(eval_file)
        datasets['test'] = eval_data
        datasets['train'] = []  # Ensure 'train' key exists even if empty
    else:
        # Mirror test expectation: raise when neither provided
        raise FileNotFoundError("At least one data file (train_file or eval_file) must be provided.")

    logging.info("Data loading complete.")
    return datasets

# Example usage (for testing purposes)
if __name__ == "__main__":
    import os
    # Create dummy files for testing
    dummy_dir = "dummy_data_csv"
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)
    train_path = os.path.join(dummy_dir, "train.tsv")
    eval_path = os.path.join(dummy_dir, "eval.tsv")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("This is human.\t1\n")
        f.write("This is machine.\t0\n")
        f.write("Another human sentence.\t1\n")
        f.write("Another machine sentence.\t0\n")
        f.write("Bad row\tX\n") # Test bad label
        f.write("Too many columns\t1\tExtra\n") # Test bad columns
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write("Test human.\t1\n")
        f.write("Test machine.\t0\n")

    try:
        print("\n--- Testing Load Both Files ---")
        datasets_dict = load_data_from_tsv(
            train_file=train_path,
            eval_file=eval_path
        )
        print(f"Train samples loaded: {len(datasets_dict.get('train', []))}")
        print(f"Test samples loaded: {len(datasets_dict.get('test', []))}")
        if datasets_dict.get('train'): print(f"First train sample: {datasets_dict['train'][0]}")
        if datasets_dict.get('test'): print(f"First test sample: {datasets_dict['test'][0]}")

        print("\n--- Testing Split Functionality ---")
        datasets_dict_split = load_data_from_tsv(
            train_file=train_path,
            eval_file=None, # Trigger split
            test_size=0.5
        )
        print(f"Train split size: {len(datasets_dict_split['train'])}")
        print(f"Test split size: {len(datasets_dict_split['test'])}")
        if datasets_dict_split.get('train'): print(f"First train sample (split): {datasets_dict_split['train'][0]}")
        if datasets_dict_split.get('test'): print(f"First test sample (split): {datasets_dict_split['test'][0]}")

        print("\n--- Testing Load Eval Only ---")
        datasets_dict_eval = load_data_from_tsv(
            train_file=None,
            eval_file=eval_path
        )
        print(f"Train samples loaded: {len(datasets_dict_eval.get('train', []))}")
        print(f"Test samples loaded: {len(datasets_dict_eval.get('test', []))}")
        if datasets_dict_eval.get('test'): print(f"First test sample (eval only): {datasets_dict_eval['test'][0]}")


    finally:
        # Clean up dummy files
        print("\n--- Cleaning up dummy files ---")
        try:
            os.remove(train_path)
            os.remove(eval_path)
            os.rmdir(dummy_dir)
            print("Cleanup complete.")
        except OSError as e:
            print(f"Error during cleanup: {e}")