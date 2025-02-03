# Paraphrase - Human Sentence Classifier

This project classifies whether a sentence is human-generated or machine-generated (paraphrased) using deep learning techniques built on the Trax library.

## Overview

The classifier uses a Transformer-based architecture with support for both subword and character-level tokenization. It leverages deep learning to capture the nuances between human and machine language.

## Installation

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Data Preparation

Prepare your training and evaluation data in TSV format:
- **Training Data (`parabank.tsv`):** The first column contains human-generated sentences (positive examples) and subsequent columns contain machine-generated (paraphrased) sentences (negative examples).

  **Example:**
  ```tsv
  Human sentence 1    Paraphrased sentence 1
  Human sentence 2    Paraphrased sentence 2
  ...
  ```

- **Evaluation Data (`eval.tsv`):** Use the same format as the training data.

Place these files in the project root or update the paths in `src/poetry/trainer/problem.py` accordingly.

## Training the Model

Run the training script with:

```bash
python src/poetry/trainer/problem.py
```

The script will:
1. Load and preprocess the data.
2. Build the vocabulary (subword or character-level).
3. Define a Transformer-based classifier.
4. Train the model.
5. Save the trained model to the specified output directory.

## Evaluation

To evaluate model performance using a confusion matrix:
1. Create the following files:
   - **Input Labels (`data/input.txt`):** Contains one label per line (`paraphrase` or `not_paraphrase`).
   - **Target Labels (`data/target.txt`):** Contains the corresponding numeric labels (`1` for paraphrase, `0` for not_paraphrase).

2. Run the evaluation script:
   ```bash
   python src/confusion_matrix.py
   ```

## Notes

- Data preparation is done manually via TSV files.
- Decoding functionality is not yet implemented.
- The evaluation script computes True Positives, True Negatives, False Positives, and False Negatives.

## Future Improvements

- Implement a Trax-based decoder for interactive classification.
- Enhance the evaluation with metrics such as precision, recall, and F1-score.
- Experiment with advanced architectures and training techniques.
- Add GPU/TPU support for faster training.
- Develop a user-friendly CLI or web interface.