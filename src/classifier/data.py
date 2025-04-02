#!/usr/bin/env python3
"""
Data handling module for the Human/Paraphrase Sentence Classifier using Hugging Face libraries.
"""

from datasets import load_dataset, DatasetDict, Dataset, Value # Import Value directly
from transformers import AutoTokenizer
from typing import Dict, List, Optional

def load_and_tokenize_data(
    train_file: str,
    eval_file: Optional[str], # Made eval_file optional
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    test_size: float = 0.1 # Optional: if eval_file is None, split train_file
) -> DatasetDict:
    """
    Loads data from TSV files, tokenizes it, and returns a DatasetDict.

    Args:
        train_file: Path to the training TSV file (columns: text, label).
        eval_file: Path to the evaluation TSV file (columns: text, label).
                    If None, the train_file will be split.
        tokenizer_name: Name of the tokenizer model from Hugging Face Hub.
        max_length: Maximum sequence length for tokenization.
        test_size: Fraction of training data to use for evaluation if eval_file is None.

    Returns:
        A DatasetDict containing 'train' and 'test' (or 'validation') splits.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define function to tokenize data
    def tokenize_function(examples: Dict[str, List]) -> Dict:
        # Ensure 'text' column exists and handle potential None values if any
        texts = [str(t) if t is not None else "" for t in examples["text"]]
        # Tokenize, pad, and truncate
        return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)

    # Load data using datasets library
    # Assumes TSV files have 'text' and 'label' columns (adjust if needed)
    # Uses 'csv' format with tab separator
    data_files = {}
    if train_file:
        data_files["train"] = train_file
    if eval_file:
        data_files["test"] = eval_file # Use 'test' split name for consistency

    if not data_files:
        raise ValueError("At least one data file (train_file or eval_file) must be provided.")

    # Load dataset specifying column names if they differ from 'text', 'label'
    # For TSV: column 0 is text, column 1 is label
    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t",
        column_names=["text", "label"], # Explicitly name columns
        skiprows=0 # Assuming no header row, adjust if header exists
    )

    # If no dedicated eval file, split the training data
    if "test" not in raw_datasets and "train" in raw_datasets:
        print(f"No evaluation file provided. Splitting train data with test_size={test_size}")
        # Ensure labels are integers if they are read as strings
        # Use cast_column for robustness before splitting
        raw_datasets["train"] = raw_datasets["train"].cast_column("label", Value("int64"))
        split_dataset = raw_datasets["train"].train_test_split(test_size=test_size, stratify_by_column="label")
        raw_datasets["train"] = split_dataset["train"]
        raw_datasets["test"] = split_dataset["test"]
        print(f"Train size: {len(raw_datasets['train'])}, Test size: {len(raw_datasets['test'])}")


    # Ensure labels are integers (might be read as strings from CSV)
    # Use cast_column for robustness
    if "train" in raw_datasets:
         raw_datasets["train"] = raw_datasets["train"].cast_column("label", Value("int64"))
    if "test" in raw_datasets:
         raw_datasets["test"] = raw_datasets["test"].cast_column("label", Value("int64"))


    # Tokenize datasets
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # Remove original text column to avoid conflicts with Trainer
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    # Rename 'label' to 'labels' for Trainer compatibility
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # Set format for PyTorch
    tokenized_datasets.set_format("torch")

    print("Data loading and tokenization complete.")
    # Check if 'train' split exists before accessing features
    if "train" in tokenized_datasets:
        print(f"Dataset features: {tokenized_datasets['train'].features}")
    elif "test" in tokenized_datasets: # Fallback to test if train doesn't exist (e.g., only eval file provided)
        print(f"Dataset features (test split): {tokenized_datasets['test'].features}")


    return tokenized_datasets

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Create dummy files for testing
    import os
    if not os.path.exists("dummy_data"):
        os.makedirs("dummy_data")
    with open("dummy_data/train.tsv", "w", encoding="utf-8") as f:
        f.write("This is human.\t1\n")
        f.write("This is machine.\t0\n")
        f.write("Another human sentence.\t1\n")
        f.write("Another machine sentence.\t0\n")
    with open("dummy_data/eval.tsv", "w", encoding="utf-8") as f:
        f.write("Test human.\t1\n")
        f.write("Test machine.\t0\n")

    try:
        datasets_dict = load_and_tokenize_data(
            train_file="dummy_data/train.tsv",
            eval_file="dummy_data/eval.tsv"
        )
        print("\nSample Train Batch:")
        print(datasets_dict["train"][0])
        print("\nSample Test Batch:")
        print(datasets_dict["test"][0])

        # Test splitting
        print("\nTesting split functionality:")
        datasets_dict_split = load_and_tokenize_data(
            train_file="dummy_data/train.tsv",
            eval_file=None, # Trigger split
            test_size=0.5
        )
        print(f"Train split size: {len(datasets_dict_split['train'])}")
        print(f"Test split size: {len(datasets_dict_split['test'])}")


    finally:
        # Clean up dummy files
        os.remove("dummy_data/train.tsv")
        os.remove("dummy_data/eval.tsv")
        os.rmdir("dummy_data")