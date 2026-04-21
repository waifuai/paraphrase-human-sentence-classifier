# Human vs. Machine Sentence Classifier (OpenRouter)

This project classifies whether a sentence is human-generated or machine-generated using OpenRouter.

## Overview

The classifier sends sentences to OpenRouter:
- OpenRouter via HTTPS (default model: `openrouter/free`)

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

### 3. Configure OpenRouter API Key

1. Obtain an API key from https://openrouter.ai/.
2. Create a file named `.api-openrouter` in your home directory.
3. Paste your API key into this file and save it without extra spaces or newlines.

The script `src/classifier/model.py` will automatically look for this file, or you can set `OPENROUTER_API_KEY` in the environment.

## Data Preparation

Prepare your evaluation data in TSV format with two columns: `text` and `label` (1=human, 0=machine). See `data/eval.tsv` for an example.

## Evaluation

Run the evaluation script using the Python interpreter from your virtual environment.

**Usage:**
```bash
# Windows:
.venv/Scripts/python.exe -m scripts.evaluate --eval_file data/eval.tsv --model_name openrouter/free
# Linux/macOS:
# .venv/bin/python -m scripts.evaluate --eval_file data/eval.tsv --model_name openrouter/free
```

**Optional Arguments:**
- `--eval_file`: (Required) Path to your evaluation TSV file.
- `--model_name`: OpenRouter model (default: value from ~/.model-openrouter if present, else `openrouter/free`).
- `--output_dir`: Directory to save the evaluation results JSON (default: `results`).
- `--delay`: Optional delay between API calls to mitigate rate limits (default: 0.1s).

The script:
1) Parses CLI args
2) Loads evaluation data
3) Calls OpenRouter via `src/classifier/model.py` per sample
4) Computes metrics (accuracy, precision, recall, F1, confusion matrix)
5) Saves results to `results/evaluation_results.json`

## Project Structure

- `src/classifier/data.py`: TSV loading.
- `src/classifier/model.py`: Provider client and prompt.
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
- Store keys securely in `~/.api-openrouter` with appropriate permissions.
- To override the default model name without passing CLI flags, create a plain-text file in your home directory:
  - `~/.model-openrouter` containing the OpenRouter model name (one line).
  This takes effect automatically as the default while the CLI flag still takes precedence when provided.

## Future Improvements

- Batching
- Retries/backoff
- Prompt optimization
- Easier multi-model comparison
- Packaging with `pyproject.toml`
