#!/usr/bin/env python3
"""Leakage-safe nested CV for traditional disease-classification baselines."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def parse_spec(spec: str) -> tuple[str, Path]:
    if ":" in spec and spec.split(":", 1)[0].lower() in {"pkl", "tsv", "csv"}:
        kind, path = spec.split(":", 1)
        return kind.lower(), Path(path)
    path = Path(spec)
    kind = "pkl" if path.suffix.lower() in {".pkl", ".pickle"} else "tsv"
    return kind, path


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def normalize_labels(values: Any) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)
    labels = labels.astype(np.int64, copy=False).reshape(-1)
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError(f"Expected binary 0/1 labels, found {np.unique(labels).tolist()}")
    return labels


def table_to_matrix(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep="," if kind == "csv" else "\t")
    if "IID" in frame.columns:
        sample_ids = frame["IID"].astype(str).to_numpy()
        frame = frame.drop(columns=[name for name in ("FID", "IID") if name in frame.columns])
    elif not pd.api.types.is_numeric_dtype(frame.iloc[:, 0]):
        sample_ids = frame.iloc[:, 0].astype(str).to_numpy()
        frame = frame.iloc[:, 1:]
    else:
        sample_ids = np.arange(len(frame)).astype(str)
    return frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32), sample_ids


def load_one(feature_spec: str, label_spec: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kind, path = parse_spec(feature_spec)
    if not path.exists():
        raise FileNotFoundError(path)
    embedded_labels = None
    if kind == "pkl":
        data = load_pickle(path)
        if not isinstance(data, (tuple, list)) or not data:
            raise ValueError(f"Expected tuple/list feature PKL: {path}")
        x_obj = data[0]
        if hasattr(x_obj, "values"):
            matrix = np.asarray(x_obj.values, dtype=np.float32)
            sample_ids = np.asarray(x_obj.index.astype(str))
        else:
            matrix = np.asarray(x_obj, dtype=np.float32)
            sample_ids = np.arange(matrix.shape[0]).astype(str)
        if len(data) > 1:
            embedded_labels = data[1]
    else:
        matrix, sample_ids = table_to_matrix(path, kind)

    if label_spec:
        label_kind, label_path = parse_spec(label_spec)
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        if label_kind == "pkl":
            label_data = load_pickle(label_path)
            label_values = label_data[1] if isinstance(label_data, (tuple, list)) and len(label_data) > 1 else label_data
        else:
            labels_frame = pd.read_csv(label_path, sep="," if label_kind == "csv" else "\t")
            label_values = labels_frame.iloc[:, -1].to_numpy()
    elif embedded_labels is not None:
        label_values = embedded_labels
    else:
        raise ValueError("Feature table has no labels; provide --labels")

    labels = normalize_labels(label_values)
    if matrix.ndim != 2 or matrix.shape[0] != labels.shape[0]:
        raise ValueError(f"X/Y mismatch: X={matrix.shape}, y={labels.shape}")
    return matrix, labels, sample_ids


def load_modalities(feature_specs: list[str], label_spec: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = []
    labels_ref = None
    ids_ref = None
    for spec in feature_specs:
        matrix, labels, sample_ids = load_one(spec, label_spec)
        if labels_ref is None:
            labels_ref, ids_ref = labels, sample_ids
        elif not np.array_equal(labels, labels_ref) or not np.array_equal(sample_ids, ids_ref):
            raise ValueError(f"Labels or sample order differ for {spec}; align modalities before concatenating")
        matrices.append(matrix)
    assert labels_ref is not None and ids_ref is not None
    return np.concatenate(matrices, axis=1), labels_ref, ids_ref


def unique_in_order(indices: np.ndarray) -> np.ndarray:
    _, positions = np.unique(indices, return_index=True)
    return indices[np.sort(positions)]


def read_fold(split_dir: Path, fold: int, n_samples: int, deduplicate_train: bool) -> tuple[np.ndarray, np.ndarray]:
    train_path = split_dir / f"outer_fold_{fold}_train_IDs.txt"
    test_path = split_dir / f"outer_fold_{fold}_test_IDs.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing fold {fold} split files under {split_dir}")
    train = np.asarray([int(value) for value in train_path.read_text().split()], dtype=np.int64)
    test = np.asarray([int(value) for value in test_path.read_text().split()], dtype=np.int64)
    if deduplicate_train:
        train = unique_in_order(train)
    for name, indices in (("train", train), ("test", test)):
        if indices.size == 0 or indices.min() < 0 or indices.max() >= n_samples:
            raise ValueError(f"Invalid {name} indices for fold {fold}")
    if np.intersect1d(np.unique(train), np.unique(test)).size:
        raise ValueError(f"Train/test overlap in fold {fold}")
    return train, test


def build_estimator(method: str, seed: int, rf_jobs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    if method == "lr":
        model = LogisticRegression(
            solver="liblinear", max_iter=2000, class_weight="balanced", random_state=seed
        )
        grid = {"model__C": [0.1, 1.0, 10.0]}
        scale = StandardScaler()
    elif method == "svm":
        model = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed)
        grid = {"model__C": [0.1, 1.0, 10.0]}
        scale = StandardScaler()
    elif method == "rf":
        model = RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=rf_jobs
        )
        grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [1, 3],
        }
        scale = "passthrough"
    elif method == "adaboost":
        base = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)
        try:
            model = AdaBoostClassifier(estimator=base, random_state=seed)
        except TypeError:  # scikit-learn < 1.2
            model = AdaBoostClassifier(base_estimator=base, random_state=seed)
        grid = {"model__n_estimators": [100, 500], "model__learning_rate": [0.05, 0.1, 1.0]}
        scale = "passthrough"
    else:
        raise ValueError(method)

    # Imputation and scaling live inside GridSearchCV, so each inner-training
    # split fits its own preprocessing statistics.
    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", scale), ("model", model)]
    )
    return pipeline, grid


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def positive_probabilities(estimator: GridSearchCV, matrix: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(matrix)
    classes = list(estimator.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Fitted estimator has unexpected classes: {classes}")
    return probabilities[:, [classes.index(0), classes.index(1)]]


def parse_folds(text: str) -> list[int]:
    return [1, 2, 3, 4, 5] if text.lower() == "all" else [int(x) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", action="append", required=True, help="Repeat to concatenate aligned modalities")
    parser.add_argument("--labels")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--methods", nargs="+", default=["lr", "rf", "svm", "adaboost"], choices=["lr", "rf", "svm", "adaboost"])
    parser.add_argument("--folds", default="all")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1521024)
    parser.add_argument("--grid-jobs", type=int, default=1)
    parser.add_argument("--rf-jobs", type=int, default=1)
    parser.add_argument("--keep-duplicate-train-rows", action="store_true")
    args = parser.parse_args()
    if args.inner_folds < 2 or args.grid_jobs == 0 or args.rf_jobs == 0:
        parser.error("--inner-folds must be >=2 and job counts cannot be zero")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix, labels, sample_ids = load_modalities(args.features, args.labels)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []

    for method in args.methods:
        for fold in parse_folds(args.folds):
            train_idx, test_idx = read_fold(
                args.split_dir, fold, len(labels), not args.keep_duplicate_train_rows
            )
            estimator, param_grid = build_estimator(method, args.seed + fold, args.rf_jobs)
            inner_cv = StratifiedKFold(
                n_splits=args.inner_folds, shuffle=True, random_state=args.seed + fold
            )
            search = GridSearchCV(
                estimator,
                param_grid,
                scoring="f1",
                cv=inner_cv,
                n_jobs=args.grid_jobs,
                refit=True,
                return_train_score=False,
            )
            search.fit(matrix[train_idx], labels[train_idx])
            prediction = search.predict(matrix[test_idx])
            probability = positive_probabilities(search, matrix[test_idx])
            row = {
                "Prefix": args.prefix,
                "Method": method,
                "Fold": fold,
                "N_train": int(len(train_idx)),
                "N_test": int(len(test_idx)),
                **classification_metrics(labels[test_idx], prediction),
            }
            metric_rows.append(row)
            params_json = json.dumps(search.best_params_, sort_keys=True)
            selection_rows.append(
                {"Method": method, "Fold": fold, "InnerBestF1": search.best_score_, "BestParams": params_json}
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "Prefix": args.prefix,
                        "Method": method,
                        "Fold": fold,
                        "SampleIndex": test_idx,
                        "SampleID": sample_ids[test_idx],
                        "Y_true": labels[test_idx],
                        "Y_pred": prediction,
                        "Prob_0": probability[:, 0],
                        "Prob_1": probability[:, 1],
                        "BestParams": params_json,
                    }
                )
            )
            print(
                f"method={method} fold={fold} F1={row['F1']:.4f} "
                f"ACC={row['Accuracy']:.4f} params={params_json}",
                flush=True,
            )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(args.output_dir / f"{args.prefix}_fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output_dir / f"{args.prefix}_outer_fold_predictions.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(args.output_dir / f"{args.prefix}_selections.csv", index=False)
    summary = (
        metrics_frame.groupby("Method")[["Precision", "Recall", "F1", "Accuracy"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(args.output_dir / f"{args.prefix}_summary.csv", index=False)


if __name__ == "__main__":
    main()
