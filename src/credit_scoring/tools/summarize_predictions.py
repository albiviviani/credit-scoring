from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def summarize(pred_csv: str, prob_col: str, pred_col: str) -> dict:
    df = pd.read_csv(pred_csv)
    if prob_col not in df.columns:
        raise SystemExit(f"Missing probability column '{prob_col}' in {pred_csv}")
    if pred_col not in df.columns:
        raise SystemExit(f"Missing prediction column '{pred_col}' in {pred_csv}")

    prob = df[prob_col]
    pred = df[pred_col]

    total = len(df)
    positives = int(pred.sum())
    negatives = total - positives
    positive_rate = positives / total if total else 0.0

    return {
        "total_rows": total,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positive_rate,
        "prob_mean": float(prob.mean()),
        "prob_median": float(prob.median()),
        "prob_min": float(prob.min()),
        "prob_max": float(prob.max()),
    }


def cli():
    ap = argparse.ArgumentParser(
        description="Summarize credit-infer predictions (counts, rates, basic stats)"
    )
    ap.add_argument("--pred-csv", required=True, help="Path to pred.csv from credit-infer")
    ap.add_argument("--prob-col", default="pd_default", help="Probability column name")
    ap.add_argument("--pred-col", default="prediction", help="Binary prediction column name")
    ap.add_argument("--save-json", default=None, help="Optional path to save summary as JSON")
    args = ap.parse_args()

    summary = summarize(args.pred_csv, args.prob_col, args.pred_col)
    print(json.dumps(summary, indent=2))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    cli()

