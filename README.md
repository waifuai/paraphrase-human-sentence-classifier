# Human vs. Machine Sentence Classifier (Google Gemini API)

This project classifies whether a sentence is human-generated or machine-generated (e.g., paraphrased) using the Google Gemini API.

## Overview

The classifier sends sentences to a specified Google Gemini model (defaulting to `gemini-2.5-pro`) via the `google-generativeai` Python library. A carefully crafted prompt instructs the model to return '1' for human-written text and '0' for machine-generated text. The project includes scripts for evaluation and basic data handling.

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url> # Replace with the actual URL
cd paraphrase-human-sentence-classifier
```

### 2. Create Virtual Environment and Install Dependencies

This project uses `uv` for environment and package management. Ensure you have Python 3.8+ installed.

```bash
# Install uv globally if you haven't already (one-time setup)
# pip install uv
# or consult uv documentation for preferred installation method

# Create the virtual environment
python -m uv venv .venv

# Ensure pip is available in the venv (recommended for compatibility)
# On Windows:
# .venv/Scripts/python.exe -m ensurepip
# On Linux/macOS:
# .venv/bin/python -m ensurepip

# Install uv within the venv (recommended)
# On Windows:
# .venv/Scripts/python.exe -m pip install uv
# On Linux/macOS:
# .venv/bin/python -m pip install uv

# Install project dependencies using the venv's uv
# On Windows:
.venv/Scripts/python.exe -m uv pip install -r requirements.txt
# On Linux/macOS:
# .venv/bin/python -m uv pip install -r requirements.txt

# To run tests later, install pytest (already included in requirements.txt)
# .venv/Scripts/python.exe -m uv pip install pytest
```
*(Note: Adjust paths like `.venv/Scripts/python.exe` to `.venv/bin/python` if you are on Linux/macOS)*

### 3. Configure Google Gemini API Key

The classifier requires a Google Gemini API key.

1.  Obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Create a file named `.api-gemini` in your **home directory** (`~` on Linux/macOS, `C:\Users\<YourUsername>` on Windows).
3.  Paste your API key into this file and save it. Ensure there are no extra spaces or newlines.

The script `src/classifier/model.py` will automatically look for this file to authenticate API calls.

## Data Preparation

Prepare your evaluation data in **tab-separated value (TSV)** format with two columns: `text` and `label`.

-   **Column 1 (text):** The sentence (human or machine-generated).
-   **Column 2 (label):** The corresponding label (`1` for human, `0` for machine).

**Example (`data/eval.tsv`):**

```tsv
This is a human sentence.	1
This text was generated by a machine.	0
Another example written by a person.	1
Paraphrased content often lacks nuance.	0
```

Place your evaluation file (e.g., `eval.tsv`) inside the `data/` directory, or specify its path using the `--eval_file` argument when running the evaluation script.

## Evaluation

To evaluate the classifier's performance on your dataset, run `src/scripts/evaluate.py` using the Python interpreter from your virtual environment.

**Usage:**

```bash
# On Windows:
.venv/Scripts/python.exe -m scripts.evaluate --eval_file data/eval.tsv
# On Linux/macOS:
# .venv/bin/python -m scripts.evaluate --eval_file data/eval.tsv
```
*(or `python -m scripts.evaluate ...` if `.venv` is activated and `src` is in PYTHONPATH)*

**Optional Arguments:**

*   `--eval_file`: (Required) Path to your evaluation TSV file.
*   `--gemini_model_name`: Specify a different Gemini model (default: `gemini-2.5-pro`).
*   `--output_dir`: Directory to save the evaluation results JSON file (default: `results`).
*   `--delay`: Add a small delay (in seconds, e.g., `0.1`) between API calls to help avoid rate limits.

The script performs the following steps:
1.  Parses command-line arguments.
2.  Loads the evaluation data using `src/classifier/data.py`.
3.  Iterates through each sentence:
    *   Calls the Gemini API via `src/classifier/model.py` to get a classification ('0' or '1').
    *   Handles potential API errors or unexpected responses.
4.  Computes metrics using `scikit-learn`: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
5.  Prints the results to the console.
6.  Saves the detailed evaluation metrics to `gemini_evaluation_results.json` within the specified output directory.

## Project Structure

-   `src/classifier/data.py`: Handles loading data from TSV files.
-   `src/classifier/model.py`: Contains the logic for interacting with the Google Gemini API, including API key handling and prompt definition.
-   `src/scripts/evaluate.py`: Main script to run the evaluation process against the Gemini API.
-   `data/`: Default directory for `eval.tsv` (and optionally `train.tsv` if needed for other purposes).
-   `results/`: Default directory for saved evaluation metrics JSON file.
-   `tests/`: Contains unit tests (`pytest`).
-   `requirements.txt`: Lists Python dependencies (install via `uv pip install -r requirements.txt`).
-   `pytest.ini`: Configures `pytest` (e.g., sets PYTHONPATH).
-   `.gitignore`: Specifies intentionally untracked files for Git.
-   `README.md`: This file.
-   `LICENSE`: Contains the MIT-0 License text.
-   `plans/`: Contains planning documents (not part of the core application).
-   `lessons/`: Contains lessons learned documents (not part of the core application).


## Notes

-   Ensure the labels in your TSV files are `1` (human) and `0` (machine).
-   API calls to Gemini can incur costs and are subject to rate limits. The `--delay` argument can help mitigate rate limit issues.
-   The classification accuracy depends heavily on the chosen Gemini model and the effectiveness of the prompt in `src/classifier/model.py`. You may need to experiment with different models or prompts for optimal performance.
-   Ensure your API key is stored securely in `~/.api-gemini` and that the file has appropriate permissions.

## Future Improvements

-   **Batching:** Implement batch API calls if the Gemini API and library support it efficiently for this type of task, potentially reducing latency and cost.
-   **Error Handling:** Add more robust error handling for API calls (e.g., retries for rate limits, specific exception handling).
-   **Prompt Optimization:** Experiment with different prompt structures, few-shot examples, or temperature settings to improve accuracy.
-   **Model Selection:** Allow easier selection or comparison between different Gemini models.
-   **Packaging:** Add `pyproject.toml` for proper packaging and distribution.