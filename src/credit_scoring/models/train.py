from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from credit_scoring.schemas import TrainingConfig
from credit_scoring.data.load_data import load_openml_german_credit
from credit_scoring.features.build_features import build_preprocessor
from credit_scoring.utils.metrics import base_metrics

def get_models():
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    try:
        import lightgbm as lgb
        models["lgbm"] = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary", class_weight="balanced", random_state=42, n_jobs=-1
        )
    except Exception:
        pass
    return models

def run(cfg: TrainingConfig):
    X, y = load_openml_german_credit()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    pre, _, _ = build_preprocessor(X_train)
    models = get_models()

    artifacts = Path("artifacts"); artifacts.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.seed)

    best_name, best_auc, best_pipe = None, -np.inf, None
    for name, clf in models.items():
        if name not in cfg.model_candidates:
            continue
        pipe = ImbPipeline(steps=[("pre", pre), ("smote", SMOTE(random_state=cfg.seed)), ("clf", clf)])
        auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        print(f"{name}: CV ROC-AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, pipe

    best_pipe.fit(X_train, y_train)
    joblib.dump({"model": best_pipe, "features": X_train.columns.tolist()}, artifacts / "model.joblib")

    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    m = base_metrics(y_test.values, y_prob)
    with open(artifacts / "metrics.json", "w") as f:
        json.dump({"cv_best_model": best_name, "cv_roc_auc": best_auc, "holdout": m}, f, indent=2)
    print(f"Best: {best_name} | CV ROC-AUC={best_auc:.4f} | Holdout ROC-AUC={m['roc_auc']:.4f}")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="openml")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--models", nargs="*", default=["logreg","rf","lgbm"])
    args = ap.parse_args()
    cfg = TrainingConfig(
        data_source=args.data_source,
        test_size=args.test_size,
        seed=args.seed,
        cv_splits=args.cv_splits,
        model_candidates=args.models,
    )
    run(cfg)

if __name__ == "__main__":
    cli()
