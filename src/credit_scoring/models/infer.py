from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import joblib
from credit_scoring.schemas import InferConfig, RecordsPayload, validate_required_columns

def run(cfg: InferConfig):
    bundle = joblib.load(cfg.model_path)
    model = bundle["model"]
    cols = bundle["features"]

    df = pd.read_csv(cfg.input_csv)
    RecordsPayload(records=df.to_dict(orient="records"))
    validate_required_columns(df.columns.tolist(), cols)

    proba = model.predict_proba(df[cols])[:, 1]
    threshold = cfg.threshold
    if threshold is None:
        try:
            import json
            with open("artifacts/metrics.json") as f:
                threshold = float(json.load(f)["holdout"]["threshold"])
        except Exception:
            threshold = 0.5
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["pd_default"] = proba
    out["prediction"] = pred
    Path(cfg.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.output_csv, index=False)
    print(f"Saved: {cfg.output_csv}")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    cfg = InferConfig(**vars(args))
    run(cfg)

if __name__ == "__main__":
    cli()
