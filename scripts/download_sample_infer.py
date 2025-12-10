from __future__ import annotations
import argparse
from pathlib import Path

from credit_scoring.data.load_data import load_openml_german_credit


def main():
    ap = argparse.ArgumentParser(
        description="Download German Credit data and save a sample CSV for inference"
    )
    ap.add_argument(
        "--output",
        default="artifacts/sample_infer.csv",
        help="Where to save the sample CSV",
    )
    ap.add_argument(
        "--n-rows",
        type=int,
        default=200,
        help="Number of rows to save (max limited by dataset size)",
    )
    args = ap.parse_args()

    X, _ = load_openml_german_credit()
    n = min(args.n_rows, len(X))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    X.head(n).to_csv(out_path, index=False)
    print(f"Saved {n} rows to {out_path}")


if __name__ == "__main__":
    main()

