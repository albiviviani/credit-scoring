from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split

from credit_scoring.schemas import EvaluateConfig
from credit_scoring.data.load_data import load_openml_german_credit
from credit_scoring.utils.metrics import base_metrics

def run(cfg: EvaluateConfig):
    bundle = joblib.load(cfg.model_path)
    model = bundle["model"]

    X, y = load_openml_german_credit()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)

    y_prob = model.predict_proba(X_te)[:, 1]
    m = base_metrics(y_te.values, y_prob)
    print(m)

    Path("artifacts").mkdir(exist_ok=True, parents=True)
    with open(cfg.metrics_path, "w") as f:
        json.dump({"holdout": m}, f, indent=2)

    fpr, tpr, _ = roc_curve(y_te, y_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("ROC"); plt.tight_bitmap = True
    plt.tight_layout(); plt.savefig("artifacts/roc.png"); plt.close()

    p, r, _ = precision_recall_curve(y_te, y_prob)
    PrecisionRecallDisplay(precision=p, recall=r).plot()
    plt.title("PR"); plt.tight_layout(); plt.savefig("artifacts/pr.png"); plt.close()

    y_pred = (y_prob >= m["threshold"]).astype(int)
    cm = confusion_matrix(y_te, y_pred, labels=[0,1])
    np.savetxt("artifacts/confusion.csv", cm, delimiter=",", fmt="%d")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--metrics-path", default="artifacts/metrics.json")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = EvaluateConfig(**vars(args))
    run(cfg)

if __name__ == "__main__":
    cli()
