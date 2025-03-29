#!/usr/bin/env python3
"""
Confusion Matrix Evaluation Script for the Paraphrase Classifier.
This script reads the input and target label files and computes the confusion matrix.
"""

from pathlib import Path


def load_lines(file_path: str) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text().splitlines()


def compute_confusion_matrix(input_lines: list[str], target_lines: list[str]) -> tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for inp, target in zip(input_lines, target_lines):
        if inp == 'paraphrase':
            if target == '1':
                tp += 1
            elif target == '0':
                fn += 1
            else:
                raise ValueError(f"Invalid target value: {target}")
        elif inp == 'not_paraphrase':
            if target == '1':
                fp += 1
            elif target == '0':
                tn += 1
            else:
                raise ValueError(f"Invalid target value: {target}")
        else:
            raise ValueError(f"Invalid input value: {inp}")
    return tp, tn, fp, fn


def main():
    # TODO: Make file paths configurable (e.g., via argparse)
    input_lines = load_lines('data/input.txt')
    target_lines = load_lines('data/target.txt')
    tp, tn, fp, fn = compute_confusion_matrix(input_lines, target_lines)
    print("Confusion Matrix:")
    print("  True Positives: ", tp)
    print("  True Negatives: ", tn)
    print("  False Positives:", fp)
    print("  False Negatives:", fn)


if __name__ == "__main__":
    main()