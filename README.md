
# Credit Scoring

An installable Python package for training and serving a credit default prediction model using the **German Credit dataset** (OpenML ID: 31). Includes CLI tools for **training**, **evaluation**, and **batch inference** with validated inputs.

## 🗂️ Project structure
```text
credit-scoring/
├─ README.md
├─ setup.cfg
├─ pyproject.toml
├─ scripts/
│  └─ download_sample_infer.py
├─ src/credit_scoring/
│  ├─ data/
│  │  └─ load_data.py
│  ├─ features/
│  │  └─ build_features.py
│  ├─ models/
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  └─ infer.py
│  ├─ tools/
│  │  └─ summarize_predictions.py
│  ├─ utils/
│  │  ├─ logger.py
│  │  └─ metrics.py
│  ├─ schemas.py
│  └─ version.py
└─ artifacts/ (created after running commands)
   ├─ model.joblib
   ├─ metrics.json
   ├─ roc.png
   ├─ pr.png
   ├─ confusion.csv
   ├─ pred.csv
   └─ pred_summary.json
```

## ✨ Features
- **Data Loading**: Downloads German Credit dataset and binarizes target (`bad` → 1, else 0).
- **Feature Engineering**: Splits numeric vs categorical columns, scales numerics, one-hot encodes categoricals.
- **Model Training** (`credit-train`): Builds preprocessing + SMOTE + model pipelines (Logistic Regression, Random Forest, optional LightGBM), performs cross-validation, saves best model.
- **Evaluation** (`credit-eval`): Reloads model, evaluates on holdout set, saves metrics and plots.
- **Inference** (`credit-infer`): Batch predictions on CSV input with schema validation.
- **Metrics**: ROC-AUC, Average Precision, F1; threshold auto-selected via PR-curve F1 max.
- **Schemas**: Pydantic-based validation for CLI inputs and payloads.

## ✅ Requirements
- Python **3.9–3.11**
- Internet access (first run downloads dataset)
- Windows PowerShell examples shown; adjust for other shells.

## 📦 Installation
```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# or
. .\.venv\Scripts\Activate.ps1  # Windows PowerShell

python -m pip install -U pip setuptools wheel
pip install -e .
```

## 🚀 Quickstart
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

## 📂 Sample Data for Inference
If you need a ready CSV with the right feature columns:
```powershell
python scripts/download_sample_infer.py --output artifacts/sample_infer.csv --n-rows 200
credit-infer --model-path artifacts/model.joblib --input-csv artifacts/sample_infer.csv --output-csv artifacts/pred.csv

```

## 📈 Summarize predictions
After running inference, you can get quick counts and probability stats:
```powershell
credit-summarize --pred-csv artifacts\pred.csv --save-json artifacts\pred_summary.json
```
This prints totals, positive rate, and prob stats; optionally saves JSON.

## 📜 Outputs
- `artifacts/model.joblib`: fitted pipeline + feature list.
- `artifacts/metrics.json`: CV winner, CV ROC-AUC, holdout metrics (ROC-AUC, AP, F1, threshold).
- `artifacts/roc.png`, `artifacts/pr.png`: evaluation plots.
- `artifacts/confusion.csv`: confusion matrix at chosen threshold.
- `artifacts/pred.csv`: (from inference) original rows + `pd_default` (probability) + `prediction` (0/1).
- `logs/credit_scoring.log`: pipeline logs for train/eval/infer.

## 🔢 Data & features
- Source: OpenML German Credit (id=31).
- Target: binary default flag (`bad`/1 mapped to 1, else 0).
- Preprocessing: StandardScaler (numerics), OneHotEncoder (categoricals), SMOTE inside pipeline before classifier.

## 📊 Models tried
- Logistic Regression (`logreg`)
- Random Forest (`rf`)
- LightGBM (`lgbm`) if LightGBM is installed and importable

## 🧪 Common options
- `--test-size` (train/eval): default 0.2
- `--seed`: default 42
- `--cv-splits` (train): default 5
- `--models` (train): subset of `logreg rf lgbm`
- `--threshold` (infer): optional; defaults to best F1 threshold saved in `metrics.json`, else 0.5
