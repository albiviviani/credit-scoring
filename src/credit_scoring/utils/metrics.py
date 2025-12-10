from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

def base_metrics(y_true, y_prob, threshold=None):
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    if threshold is None:
        p, r, t = precision_recall_curve(y_true, y_prob)
        f1s = 2 * (p*r) / np.clip(p + r, 1e-9, None)
        idx = np.nanargmax(f1s)
        threshold = 0.5 if idx >= len(t) else t[idx]
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    return {"roc_auc": float(auc), "avg_precision": float(ap), "f1": float(f1), "threshold": float(threshold)}
