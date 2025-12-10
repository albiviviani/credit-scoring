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
from credit_scoring.utils.logger import get_logger

logger = get_logger(__name__)

def run(cfg: EvaluateConfig):
    logger.info("Starting evaluation with config: %s", cfg.model_dump())
    bundle = joblib.load(cfg.model_path)
    model = bundle["model"]
    logger.info("Loaded model bundle from %s", cfg.model_path)

    X, y = load_openml_german_credit()
    logger.info("Loaded dataset: X=%s, y=%s, positive_rate=%.3f", X.shape, y.shape, y.mean())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    logger.info("Split data: train=%d, test=%d, test_size=%.2f", len(X_tr), len(X_te), cfg.test_size)

    y_prob = model.predict_proba(X_te)[:, 1]
    m = base_metrics(y_te.values, y_prob)
    logger.info(
        "Holdout metrics | ROC-AUC=%.4f | AP=%.4f | F1=%.4f | threshold=%.3f",
        m["roc_auc"], m["avg_precision"], m["f1"], m["threshold"]
    )

    Path("artifacts").mkdir(exist_ok=True, parents=True)
    with open(cfg.metrics_path, "w") as f:
        json.dump({"holdout": m}, f, indent=2)
    logger.info("Saved metrics to %s", cfg.metrics_path)

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
    logger.info("Saved plots and confusion matrix to artifacts/")

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
