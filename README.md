# Credit Scoring

Installable Python package that trains and serves a credit default model on the German Credit dataset (OpenML id=31). Provides CLI commands for training, evaluation, and batch inference with validated inputs.

## What’s inside
- `credit_scoring/data/load_data.py`: downloads German Credit from OpenML (id=31) and binarizes target (`bad`/1 -> 1, else 0).
- `credit_scoring/features/build_features.py`: splits numeric vs categorical columns; scales numerics; one-hot encodes categoricals.
- `credit_scoring/models/train.py` (`credit-train`): builds preprocessing + SMOTE + model pipelines (logreg, random forest, optional LightGBM), cross-validates, fits best, saves artifacts.
- `credit_scoring/models/evaluate.py` (`credit-eval`): reloads model, evaluates on fresh split, saves metrics and plots.
- `credit_scoring/models/infer.py` (`credit-infer`): batch inference on a CSV; validates required columns; writes predictions.
- `credit_scoring/utils/metrics.py`: ROC-AUC, average precision, F1; auto-threshold selected via PR-curve F1 max.
- `credit_scoring/schemas.py`: Pydantic config/validation for CLI inputs and payloads.

## Requirements
- Python 3.9–3.11
- Internet access on first run (downloads OpenML dataset)
- Windows PowerShell commands shown; adjust paths for other shells.

## Quickstart (Windows PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -e .

# Train (downloads data, runs CV, saves artifacts/)
credit-train --test-size 0.2 --seed 42

# Evaluate a saved model
credit-eval --model-path artifacts\model.joblib --metrics-path artifacts\metrics.json

# Batch inference on your CSV
credit-infer --model-path artifacts\model.joblib --input-csv path\to\your.csv --output-csv artifacts\pred.csv
```

### Get sample data for inference
If you need a ready CSV with the right feature columns:
```powershell
python scripts\download_sample_infer.py --output artifacts\sample_infer.csv --n-rows 200
credit-infer --model-path artifacts\model.joblib --input-csv artifacts\sample_infer.csv --output-csv artifacts\pred.csv
```

### Summarize predictions
After running inference, you can get quick counts and probability stats:
```powershell
credit-summarize --pred-csv artifacts\pred.csv --save-json artifacts\pred_summary.json
```
This prints totals, positive rate, and prob stats; optionally saves JSON.

## Outputs
- `artifacts/model.joblib`: fitted pipeline + feature list.
- `artifacts/metrics.json`: CV winner, CV ROC-AUC, holdout metrics (ROC-AUC, AP, F1, threshold).
- `artifacts/roc.png`, `artifacts/pr.png`: evaluation plots.
- `artifacts/confusion.csv`: confusion matrix at chosen threshold.
- `artifacts/pred.csv`: (from inference) original rows + `pd_default` (probability) + `prediction` (0/1).

## Data & features
- Source: OpenML German Credit (id=31).
- Target: binary default flag (`bad`/1 mapped to 1, else 0).
- Preprocessing: StandardScaler (numerics), OneHotEncoder (categoricals), SMOTE inside pipeline before classifier.

## Models tried
- Logistic Regression (`logreg`)
- Random Forest (`rf`)
- LightGBM (`lgbm`) if LightGBM is installed and importable

## Common options
- `--test-size` (train/eval): default 0.2
- `--seed`: default 42
- `--cv-splits` (train): default 5
- `--models` (train): subset of `logreg rf lgbm`
- `--threshold` (infer): optional; defaults to best F1 threshold saved in `metrics.json`, else 0.5

## Project structure (selected)
- `setup.cfg`, `pyproject.toml`: packaging and entry points (`credit-train`, `credit-eval`, `credit-infer`)
- `src/credit_scoring/`: code
- `artifacts/`: generated models, metrics, plots (created after running commands)

## Troubleshooting
- NumPy 2.x breaks OpenML (`np.sctypes` removed). Pinned to `numpy==1.26.4` in setup; if you upgraded, reinstall: `pip install numpy==1.26.4`.
- If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
- If LightGBM fails to build on Windows, the code skips it automatically; training still runs with other models.
