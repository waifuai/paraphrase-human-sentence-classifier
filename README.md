# Human vs. Machine Sentence Classifier (OpenRouter default; Google GenAI optional)

This project classifies whether a sentence is human-generated or machine-generated using OpenRouter by default, and can alternatively use the Google GenAI SDK.

## Overview

The classifier sends sentences to a specified provider:
- OpenRouter via HTTPS (default model: `openrouter/horizon-beta`)
- Google Gemini via the Google GenAI SDK (default model: `gemini-2.5-pro`)

A carefully crafted prompt instructs the model to return '1' for human-written text and '0' for machine-generated text. The project includes scripts for evaluation and basic data handling.

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url> # Replace with the actual URL
cd paraphrase-human-sentence-classifier
```

### 2. Create Virtual Environment and Install Dependencies

This project uses `uv` for environment and package management. Ensure you have Python 3.8+ installed.

```bash
# Create the virtual environment
python -m uv venv .venv

# Ensure pip is available in the venv (recommended)
# Windows:
# .venv/Scripts/python.exe -m ensurepip
# Linux/macOS:
# .venv/bin/python -m ensurepip

# Install uv inside the venv (optional but recommended)
# Windows:
# .venv/Scripts/python.exe -m pip install uv
# Linux/macOS:
# .venv/bin/python -m pip install uv

# Install project dependencies using the venv's uv
# Windows:
.venv/Scripts/python.exe -m uv pip install -r requirements.txt
# Linux/macOS:
# .venv/bin/python -m uv pip install -r requirements.txt
```

### 3. Configure Google Gemini API Key

1. Obtain an API key from Google AI Studio.
2. Create a file named `.api-gemini` in your home directory.
3. Paste your API key into this file and save it without extra spaces or newlines.

The script `src/classifier/model.py` will automatically look for this file, or you can set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment.

### 4. Configure OpenRouter API Key

1. Obtain an API key from https://openrouter.ai/.
2. Create a file named `.api-openrouter` in your home directory.
3. Paste your API key into this file and save it without extra spaces or newlines.

Alternatively, you can set the environment variable `OPENROUTER_API_KEY` and it will take precedence.

## Data Preparation

Prepare your evaluation data in TSV format with two columns: `text` and `label` (1=human, 0=machine). See `data/eval.tsv` for an example.

## Evaluation

Run the evaluation script using the Python interpreter from your virtual environment.

**Usage (Gemini default):**
```bash
# Windows:
.venv/Scripts/python.exe -m scripts.evaluate --eval_file data/eval.tsv
# Linux/macOS:
# .venv/bin/python -m scripts.evaluate --eval_file data/eval.tsv
```

To use OpenRouter instead of Gemini:
```bash
# Windows:
.venv/Scripts/python.exe -m scripts.evaluate --eval_file data/eval.tsv --provider openrouter --openrouter_model_name openrouter/horizon-beta
# Linux/macOS:
# .venv/bin/python -m scripts.evaluate --eval_file data/eval.tsv --provider openrouter --openrouter_model_name openrouter/horizon-beta
```

**Optional Arguments:**
- `--eval_file`: (Required) Path to your evaluation TSV file.
- `--provider`: Choose `gemini` or `openrouter` (default: `gemini`).
- `--openrouter_model_name`: OpenRouter model (default: value from ~/.model-openrouter if present, else `openrouter/horizon-beta`).
- `--gemini_model_name`: Gemini model (default: value from ~/.model-gemini if present, else `gemini-2.5-pro`).
- `--output_dir`: Directory to save the evaluation results JSON (default: `results`).
- `--delay`: Optional delay between API calls to mitigate rate limits (default: 0.1s).

The script:
1) Parses CLI args
2) Loads evaluation data
3) Calls the selected provider via `src/classifier/model.py` per sample
4) Computes metrics (accuracy, precision, recall, F1, confusion matrix)
5) Saves results to `results/gemini_evaluation_results.json`

## Project Structure

- `src/classifier/data.py`: TSV loading.
- `src/classifier/model.py`: Provider clients and prompt.
- `src/scripts/evaluate.py`: Evaluation CLI.
- `data/`: Dataset samples.
- `results/`: Output metrics.
- `tests/`: Unit tests.
- `requirements.txt`: Dependencies.
- `pytest.ini`: Pytest config.
- `.gitignore`, `README.md`, `LICENSE`.

## Notes

- Costs and rate limits apply. Use `--delay` to mitigate.
- Experiment with models and prompt for accuracy.
- Store keys securely in `~/.api-gemini` and `~/.api-openrouter` with appropriate permissions.
- To override default model names without passing CLI flags, create plain-text files in your home directory:
  - `~/.model-gemini` containing the Gemini model name (one line).
  - `~/.model-openrouter` containing the OpenRouter model name (one line).
  These take effect automatically as defaults while CLI flags still take precedence when provided.

## Future Improvements

- Batching
- Retries/backoff
- Prompt optimization
- Easier multi-model comparison
- Packaging with `pyproject.toml`