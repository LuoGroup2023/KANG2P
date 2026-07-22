#!/usr/bin/env python3
"""Aggregate DualKAN per-task outputs from the parallel GPU launcher."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["R2", "MSE", "PCC", "SCC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DualKAN fold metrics and predictions")
    parser.add_argument("run_dir", type=Path, help="Root output directory produced by run_dualkan_all_crops_caduceus.sh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    metric_files = sorted(p for p in run_dir.rglob("dualkan_fold_metrics.csv") if "outer_fold_" not in str(p))
    prediction_files = sorted(p for p in run_dir.rglob("dualkan_predictions.csv") if "outer_fold_" not in str(p))

    if not metric_files:
        raise FileNotFoundError(f"No dualkan_fold_metrics.csv files found under {run_dir}")

    metrics = []
    for path in metric_files:
        df = pd.read_csv(path)
        df.insert(0, "Source_dir", str(path.parent.relative_to(run_dir)))
        metrics.append(df)
    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_df.to_csv(run_dir / "dualkan_all_fold_metrics.csv", index=False)

    summary = (
        metrics_df.groupby(["Dataset", "Trait"], as_index=False)[METRICS]
        .agg(["mean", "std", "median", "min", "max"])
    )
    summary.columns = [
        "_".join([str(x) for x in col if x]) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(run_dir / "dualkan_all_summary.csv", index=False)

    if prediction_files:
        preds = []
        for path in prediction_files:
            df = pd.read_csv(path)
            df.insert(0, "Source_dir", str(path.parent.relative_to(run_dir)))
            preds.append(df)
        pd.concat(preds, ignore_index=True).to_csv(run_dir / "dualkan_all_predictions.csv", index=False)

    print(f"Wrote {run_dir / 'dualkan_all_fold_metrics.csv'}")
    print(f"Wrote {run_dir / 'dualkan_all_summary.csv'}")
    if prediction_files:
        print(f"Wrote {run_dir / 'dualkan_all_predictions.csv'}")


if __name__ == "__main__":
    main()
