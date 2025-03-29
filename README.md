# Paraphrase - Human Sentence Classifier

This project classifies whether a sentence is human-generated or machine-generated (paraphrased) using deep learning techniques built on the Trax library.

## Overview

The classifier uses a simple embedding-based model (configurable) with support for both subword and character-level tokenization. It leverages deep learning to capture the nuances between human and machine language. The project structure has been refactored for better modularity and maintainability.

## Installation

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```
*(Note: Ensure `src` is in your PYTHONPATH if running scripts from the root directory, as configured in `pytest.ini`)*

## Data Preparation

Prepare your training and evaluation data in TSV format:
- **Training Data (`parabank.tsv` by default):** The first column contains human-generated sentences (positive examples) and subsequent columns contain machine-generated (paraphrased) sentences (negative examples).

  **Example:**
  ```tsv
  Human sentence 1    Paraphrased sentence 1
  Human sentence 2    Paraphrased sentence 2
  ...
  ```

- **Evaluation Data (`eval.tsv` by default):** Use the same format as the training data.

Place these files in the project root or specify their paths using command-line arguments when running the training script.

## Training the Model

Run the training script `src/scripts/train.py`. Configuration is handled via command-line arguments.

**Basic Usage:**
```bash
python src/scripts/train.py
```

This will use default settings (e.g., `parabank.tsv` for training, `eval.tsv` for evaluation, `subword` vocabulary, output to `model_output/`).

**Custom Configuration:**
You can override defaults using arguments:
```bash
python src/scripts/train.py \
    --train_file path/to/your/train_data.tsv \
    --eval_file path/to/your/eval_data.tsv \
    --output_dir custom_model_output/ \
    --vocab_type character \
    --batch_size 32 \
    --n_steps 1000
```
Run `python src/scripts/train.py --help` for a full list of options.

The script performs the following steps:
1. Parses command-line arguments for configuration.
2. Loads and preprocesses the training and evaluation data using `src/poetry/data.py`.
3. Builds the vocabulary (subword or character-level).
4. Initializes the classifier model defined in `src/poetry/model.py`.
5. Trains the model using Trax's training loop.
6. Saves checkpoints and the final model to the specified output directory.

## Evaluation

The evaluation script `src/scripts/evaluate.py` computes a confusion matrix based on predefined input and target files.

**Note:** This script currently uses hardcoded paths `data/input.txt` and `data/target.txt`. This needs refactoring to be more flexible (e.g., take paths as arguments or evaluate a saved model checkpoint).

1. Create the following files (if using the current hardcoded paths):
   - **Input Labels (`data/input.txt`):** Contains one predicted label per line (`paraphrase` or `not_paraphrase`). *This file format seems designed for post-prediction analysis rather than direct model evaluation.*
   - **Target Labels (`data/target.txt`):** Contains the corresponding true numeric labels (`1` for paraphrase, `0` for not_paraphrase).

2. Run the evaluation script:
   ```bash
   python src/scripts/evaluate.py
   ```

## Project Structure

- `src/poetry/data.py`: Handles data loading, preprocessing, vocabulary building, and batch generation.
- `src/poetry/model.py`: Defines the classifier model architecture.
- `src/scripts/train.py`: Main script to run the training process. Configurable via command-line arguments.
- `src/scripts/evaluate.py`: Script to compute the confusion matrix (currently uses hardcoded paths).
- `data/`: Default directory for data files (though paths are configurable).
- `model_output/`: Default directory for saved models and checkpoints.
- `tests/`: Contains unit/integration tests.
- `plans/`: Contains planning documents.

## Notes

- Data preparation relies on specific TSV formatting.
- The current evaluation script (`evaluate.py`) is basic and needs improvement for flexibility and integration with the trained model.
- Decoding/prediction functionality for classifying new sentences is not yet implemented.

## Future Improvements (From Original README & Refactoring)

- **Enhance Evaluation:**
    - Modify `evaluate.py` to load a trained model checkpoint.
    - Evaluate directly on the evaluation dataset (`eval.tsv` or specified file).
    - Compute and report standard metrics (Precision, Recall, F1-score) in addition to the confusion matrix.
    - Make input/target paths configurable.
- **Implement Prediction/Decoding:** Create a script or function to load a trained model and classify new, unseen sentences interactively or from a file.
- **Configuration:** Add more command-line arguments to `train.py` (e.g., learning rate, model dimensions, device selection). Consider using a config file (YAML/JSON) for complex configurations.
- **Model Architecture:** Experiment with more advanced architectures (e.g., full Transformer encoder) within `model.py`.
- **GPU/TPU Support:** Add explicit configuration for training on different devices.
- **Testing:** Expand test coverage, especially for the data pipeline and training script logic.
- **Interface:** Develop a user-friendly CLI or web interface for interaction.
- **Dependency Management:** Consider fully adopting Poetry (`pyproject.toml`) for dependency management and packaging.