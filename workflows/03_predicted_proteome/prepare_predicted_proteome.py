#!/usr/bin/env python3
"""Align, filter, impute, and optionally standardize predicted-proteome tables.

This utility starts from per-sample protein predictions produced by an upstream
pQTL/protein-weight engine. It does not train or distribute those external
weight models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_ids(path: Path) -> list[str]:
    values = pd.read_csv(path, header=None).iloc[:, 0].astype(str).str.strip().tolist()
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Reference ID file must contain unique IDs: {path}")
    return values


def read_prediction_table(path: Path, sample_id_column: str | None) -> pd.DataFrame:
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    frame = pd.read_csv(path, sep=separator)
    if frame.empty:
        raise ValueError(f"Empty prediction table: {path}")

    if sample_id_column:
        if sample_id_column not in frame.columns:
            raise KeyError(f"{sample_id_column!r} is absent from {path}")
        ids = frame[sample_id_column].astype(str)
    elif "IID" in frame.columns:
        ids = frame["IID"].astype(str)
    elif "FID" in frame.columns:
        ids = frame["FID"].astype(str)
    elif not pd.api.types.is_numeric_dtype(frame.iloc[:, 0]):
        ids = frame.iloc[:, 0].astype(str)
        frame = frame.iloc[:, 1:]
    else:
        raise ValueError(f"Cannot infer sample IDs in {path}; provide --sample-id-column")

    frame = frame.drop(columns=[name for name in ("FID", "IID", sample_id_column) if name and name in frame.columns])
    frame.index = ids.str.strip()
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate sample IDs in {path}")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric.columns = numeric.columns.astype(str)
    return numeric


def align_tables(tables: list[pd.DataFrame], source_paths: list[Path]) -> pd.DataFrame:
    reference_ids = tables[0].index
    aligned = []
    used_columns: set[str] = set()
    for table, source in zip(tables, source_paths):
        missing = reference_ids.difference(table.index)
        extra = table.index.difference(reference_ids)
        if len(missing) or len(extra):
            raise ValueError(
                f"Sample set mismatch for {source}: missing={len(missing)}, extra={len(extra)}"
            )
        table = table.loc[reference_ids].copy()
        overlap = used_columns.intersection(table.columns)
        if overlap:
            raise ValueError(f"Duplicate protein columns across tables; first examples={sorted(overlap)[:5]}")
        used_columns.update(table.columns)
        aligned.append(table)
    return pd.concat(aligned, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-table", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id-column")
    parser.add_argument("--keep-features", type=Path, help="Optional one-column protein/model allow-list")
    parser.add_argument(
        "--fit-sample-ids",
        type=Path,
        help="IDs used to estimate imputation/scaling values. Required with --standardize.",
    )
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--max-missing-fraction", type=float, default=0.2)
    parser.add_argument("--min-std", type=float, default=1e-8)
    args = parser.parse_args()
    if not 0.0 <= args.max_missing_fraction <= 1.0:
        parser.error("--max-missing-fraction must be in [0, 1]")
    if args.standardize and args.fit_sample_ids is None:
        parser.error("--standardize requires --fit-sample-ids to avoid whole-cohort leakage")
    return args


def main() -> None:
    args = parse_args()
    for path in args.prediction_table:
        if not path.exists():
            raise FileNotFoundError(path)

    tables = [read_prediction_table(path, args.sample_id_column) for path in args.prediction_table]
    frame = align_tables(tables, args.prediction_table)
    n_input_features = frame.shape[1]

    if args.keep_features:
        keep = pd.read_csv(args.keep_features, header=None).iloc[:, 0].astype(str).tolist()
        missing_requested = sorted(set(keep).difference(frame.columns))
        if missing_requested:
            raise ValueError(f"Allow-list proteins absent from predictions: {missing_requested[:10]}")
        frame = frame.loc[:, keep]

    fit_ids = read_ids(args.fit_sample_ids) if args.fit_sample_ids else frame.index.tolist()
    missing_fit_ids = sorted(set(fit_ids).difference(frame.index))
    if missing_fit_ids:
        raise ValueError(f"Fit IDs absent from predictions: {missing_fit_ids[:10]}")
    fit = frame.loc[fit_ids]

    missing_fraction = fit.isna().mean(axis=0)
    std = fit.std(axis=0, ddof=0)
    keep_mask = (missing_fraction <= args.max_missing_fraction) & np.isfinite(std) & (std > args.min_std)
    dropped = pd.DataFrame(
        {
            "feature": frame.columns[~keep_mask],
            "fit_missing_fraction": missing_fraction[~keep_mask].to_numpy(),
            "fit_std": std[~keep_mask].to_numpy(),
        }
    )
    frame = frame.loc[:, keep_mask]
    fit = fit.loc[:, keep_mask]

    medians = fit.median(axis=0).fillna(0.0)
    frame = frame.fillna(medians)
    means = frame.loc[fit_ids].mean(axis=0)
    scales = frame.loc[fit_ids].std(axis=0, ddof=0).replace(0.0, 1.0)
    if args.standardize:
        frame = (frame - means) / scales

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export = frame.copy()
    export.insert(0, "IID", export.index)
    export.insert(0, "FID", export.index)
    export.to_csv(args.output, sep="\t", index=False)
    dropped.to_csv(args.output.with_suffix(args.output.suffix + ".dropped_features.tsv"), sep="\t", index=False)
    pd.DataFrame(
        {
            "feature": frame.columns,
            "fit_mean": means[frame.columns].to_numpy(),
            "fit_scale": scales[frame.columns].to_numpy(),
            "fit_median": medians[frame.columns].to_numpy(),
        }
    ).to_csv(args.output.with_suffix(args.output.suffix + ".transform.tsv"), sep="\t", index=False)

    config = {
        "prediction_tables": [str(path) for path in args.prediction_table],
        "output": str(args.output),
        "n_samples": int(frame.shape[0]),
        "n_input_features": int(n_input_features),
        "n_output_features": int(frame.shape[1]),
        "n_dropped_features": int(len(dropped)),
        "standardized": bool(args.standardize),
        "fit_sample_ids": str(args.fit_sample_ids) if args.fit_sample_ids else "all samples (imputation/QC only)",
    }
    args.output.with_suffix(args.output.suffix + ".config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
