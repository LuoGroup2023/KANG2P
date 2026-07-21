#!/usr/bin/env python3
"""Fold-safe nested cross-validation for the DiseaseCapsule baseline.

The script accepts either a PKL feature bundle or a tabular feature matrix and
uses the same implementation for genotype (G), predicted expression (PE),
predicted protein abundance (PP), and concatenated feature sets.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
    raise SystemExit("optuna is required; install requirements/disease.txt") from exc


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def normalize_labels(values: Any) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    y = y.astype(np.int64, copy=False).reshape(-1)
    labels = np.unique(y)
    if not np.array_equal(labels, np.array([0, 1])):
        raise ValueError(f"Expected binary labels encoded as 0/1; found {labels.tolist()}")
    return y


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


def table_to_matrix(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    sep = "," if kind == "csv" else "\t"
    frame = pd.read_csv(path, sep=sep)
    if frame.empty:
        raise ValueError(f"Empty feature table: {path}")

    if "IID" in frame.columns:
        sample_ids = frame["IID"].astype(str).to_numpy()
        drop_cols = [name for name in ("FID", "IID") if name in frame.columns]
        frame = frame.drop(columns=drop_cols)
    elif not pd.api.types.is_numeric_dtype(frame.iloc[:, 0]):
        sample_ids = frame.iloc[:, 0].astype(str).to_numpy()
        frame = frame.iloc[:, 1:]
    else:
        sample_ids = np.arange(len(frame)).astype(str)

    numeric = frame.apply(pd.to_numeric, errors="raise")
    return numeric.to_numpy(dtype=np.float32), sample_ids


def load_dataset(feature_spec: str, label_spec: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kind, path = parse_spec(feature_spec)
    if not path.exists():
        raise FileNotFoundError(path)

    embedded_y = None
    if kind == "pkl":
        data = load_pickle(path)
        if not isinstance(data, (tuple, list)) or len(data) < 1:
            raise ValueError(f"Expected tuple/list PKL feature bundle: {path}")
        x_obj = data[0]
        if hasattr(x_obj, "values"):
            x = np.asarray(x_obj.values, dtype=np.float32)
            sample_ids = np.asarray(x_obj.index.astype(str))
        else:
            x = np.asarray(x_obj, dtype=np.float32)
            sample_ids = np.arange(x.shape[0]).astype(str)
        if len(data) > 1:
            embedded_y = data[1]
    else:
        x, sample_ids = table_to_matrix(path, kind)

    if label_spec:
        label_kind, label_path = parse_spec(label_spec)
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        if label_kind == "pkl":
            label_data = load_pickle(label_path)
            y_obj = label_data[1] if isinstance(label_data, (tuple, list)) and len(label_data) > 1 else label_data
        else:
            sep = "," if label_kind == "csv" else "\t"
            label_frame = pd.read_csv(label_path, sep=sep)
            y_obj = label_frame.iloc[:, -1].to_numpy()
    elif embedded_y is not None:
        y_obj = embedded_y
    else:
        raise ValueError("Labels are absent from the feature PKL; provide --labels.")

    y = normalize_labels(y_obj)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"X/Y mismatch: X={x.shape}, y={y.shape}")
    if sample_ids.shape[0] != x.shape[0]:
        raise ValueError("Sample ID count does not match the feature matrix.")
    return x, y, sample_ids


def load_modalities(feature_specs: list[str], label_spec: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = []
    reference_y = None
    reference_ids = None
    for spec in feature_specs:
        matrix, labels, sample_ids = load_dataset(spec, label_spec)
        if reference_y is None:
            reference_y = labels
            reference_ids = sample_ids
        else:
            if not np.array_equal(labels, reference_y):
                raise ValueError(f"Label order differs for feature specification {spec}")
            if not np.array_equal(sample_ids, reference_ids):
                raise ValueError(
                    f"Sample order differs for {spec}; align the table explicitly before multimodal concatenation"
                )
        matrices.append(matrix)
    assert reference_y is not None and reference_ids is not None
    return np.concatenate(matrices, axis=1), reference_y, reference_ids


def unique_in_order(indices: np.ndarray) -> np.ndarray:
    _, positions = np.unique(indices, return_index=True)
    return indices[np.sort(positions)]


def read_fold_indices(split_dir: Path, fold: int, n_samples: int, deduplicate_train: bool) -> tuple[np.ndarray, np.ndarray]:
    train_path = split_dir / f"outer_fold_{fold}_train_IDs.txt"
    test_path = split_dir / f"outer_fold_{fold}_test_IDs.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing outer-fold files for fold {fold} under {split_dir}")
    train = np.asarray([int(value) for value in train_path.read_text().split()], dtype=np.int64)
    test = np.asarray([int(value) for value in test_path.read_text().split()], dtype=np.int64)
    if deduplicate_train:
        train = unique_in_order(train)
    for name, values in (("train", train), ("test", test)):
        if values.size == 0 or values.min() < 0 or values.max() >= n_samples:
            raise ValueError(f"Invalid {name} indices in outer fold {fold}")
    overlap = np.intersect1d(np.unique(train), np.unique(test))
    if overlap.size:
        raise ValueError(f"Outer fold {fold} has {overlap.size} train/test sample overlaps")
    return train, test


def fit_transform_train(train: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(train, axis=0)
    means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
    train = np.where(np.isfinite(train), train, means)
    target = np.where(np.isfinite(target), target, means)
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train).astype(np.float32),
        scaler.transform(target).astype(np.float32),
    )


class ConvCaps2D(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.primary_capslen = int(config["primary_capslen"])
        self.capsules = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.primary_capslen,
                    kernel_size=(1, int(config["kernel_size"])),
                    stride=int(config["stride"]),
                )
                for _ in range(int(config["filters"]))
            ]
        )

    @staticmethod
    def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        norm = (tensor**2).sum(dim=dim, keepdim=True)
        return norm / (1.0 + norm) * tensor / torch.sqrt(norm + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [capsule(x).reshape(x.size(0), self.primary_capslen, -1) for capsule in self.capsules]
        return self.squash(torch.cat(outputs, dim=2).permute(0, 2, 1))


class DigitCaps(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.num_iterations = int(config["routing_iterations"])
        self.num_caps = 2
        neurons = int(config["neurons"])
        kernel_size = int(config["kernel_size"])
        stride = int(config["stride"])
        filters = int(config["filters"])
        self.in_channels = int(config["primary_capslen"])
        self.out_channels = int(config["digit_capslen"])
        self.num_routes = ((neurons - kernel_size) // stride + 1) * filters
        self.weight = nn.Parameter(
            torch.randn(self.num_caps, self.num_routes, self.in_channels, self.out_channels)
        )

    @staticmethod
    def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        norm = (tensor**2).sum(dim=dim, keepdim=True)
        return norm / (1.0 + norm) * tensor / torch.sqrt(norm + 1e-8)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        votes = torch.matmul(u[:, None, :, None, :], self.weight)
        logits = torch.zeros_like(votes)
        for iteration in range(self.num_iterations):
            couplings = F.softmax(logits.transpose(2, -1), dim=-1).transpose(2, -1)
            output = self.squash((couplings * votes).sum(dim=2, keepdim=True))
            if iteration + 1 < self.num_iterations:
                logits = logits + (votes * output).sum(dim=-1, keepdim=True)
        output = output.reshape(output.size(0), self.num_caps, self.out_channels)
        return torch.sqrt((output**2).sum(dim=-1) + 1e-8)


class CapsNet(nn.Module):
    def __init__(self, input_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, int(config["neurons"]))
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.primary = ConvCaps2D(config)
        self.digit = DigitCaps(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.digit(self.primary(F.relu(self.dropout(self.fc(x)))))


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required in every training fold; counts={counts.tolist()}")
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(x).reshape(x.shape[0], 1, 1, x.shape[1]).float()
    y_tensor = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: CapsNet,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> CapsNet:
    loader = make_loader(x, y, batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    model.to(device)
    for _ in range(epochs):
        model.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def predict(model: CapsNet, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    dummy = np.zeros(x.shape[0], dtype=np.int64)
    loader = make_loader(x, dummy, batch_size, shuffle=False)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            probabilities.append(F.softmax(model(inputs.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def trial_config(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "neurons": trial.suggest_categorical("neurons", [150, 300, 500]),
        "dropout": trial.suggest_categorical("dropout", [0.1, 0.3, 0.5]),
        "primary_capslen": trial.suggest_categorical("primary_capslen", [4, 8, 16]),
        "digit_capslen": trial.suggest_categorical("digit_capslen", [8, 16]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "stride": trial.suggest_int("stride", 1, 3),
        "filters": trial.suggest_categorical("filters", [16, 24, 32]),
        "routing_iterations": trial.suggest_int("routing_iterations", 1, 3),
    }


DEFAULT_CONFIG: dict[str, Any] = {
    "neurons": 300,
    "dropout": 0.3,
    "primary_capslen": 8,
    "digit_capslen": 16,
    "kernel_size": 5,
    "stride": 2,
    "filters": 24,
    "routing_iterations": 2,
}


def tune_config(
    x: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    outer_fold: int,
) -> dict[str, Any]:
    if args.n_trials == 0:
        return dict(DEFAULT_CONFIG)

    def objective(trial: optuna.Trial) -> float:
        config = trial_config(trial)
        splitter = StratifiedKFold(
            n_splits=args.inner_folds, shuffle=True, random_state=args.seed + outer_fold
        )
        scores = []
        for inner_fold, (train_pos, valid_pos) in enumerate(splitter.split(x, y), start=1):
            seed_everything(args.seed + outer_fold * 10_000 + trial.number * 100 + inner_fold)
            x_train, x_valid = fit_transform_train(x[train_pos], x[valid_pos])
            model = CapsNet(x.shape[1], config)
            train_model(
                model, x_train, y[train_pos], device, args.tune_epochs, args.batch_size, args.learning_rate
            )
            probability = predict(model, x_valid, device, args.batch_size)
            scores.append(metrics(y[valid_pos], probability.argmax(axis=1))["F1"])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=args.seed + outer_fold)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    config = dict(DEFAULT_CONFIG)
    config.update(study.best_params)
    return config


def parse_folds(text: str) -> list[int]:
    if text.lower() == "all":
        return [1, 2, 3, 4, 5]
    return [int(value) for value in text.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        action="append",
        required=True,
        help="Repeatable pkl:path, tsv:path, csv:path, or inferred path; repeated matrices are concatenated.",
    )
    parser.add_argument("--labels", help="Optional label bundle; required for feature-only tables")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=15, help="Use 0 for the documented default config")
    parser.add_argument("--tune-epochs", type=int, default=15)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1521024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--keep-duplicate-train-rows",
        action="store_true",
        help="Preserve repeated outer-training indices. Default deduplicates pre-upsampled split files.",
    )
    args = parser.parse_args()
    if args.inner_folds < 2 or args.n_trials < 0:
        parser.error("--inner-folds must be >=2 and --n-trials must be >=0")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        requested = torch.device(args.device)
        device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")

    x, y, sample_ids = load_modalities(args.features, args.labels)
    fold_metrics: list[dict[str, Any]] = []
    fold_predictions: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []

    for fold in parse_folds(args.folds):
        train_idx, test_idx = read_fold_indices(
            args.split_dir, fold, len(y), deduplicate_train=not args.keep_duplicate_train_rows
        )
        config = tune_config(x[train_idx], y[train_idx], args, device, fold)
        x_train, x_test = fit_transform_train(x[train_idx], x[test_idx])
        seed_everything(args.seed + fold * 100_000)
        model = train_model(
            CapsNet(x.shape[1], config),
            x_train,
            y[train_idx],
            device,
            args.final_epochs,
            args.batch_size,
            args.learning_rate,
        )
        probability = predict(model, x_test, device, args.batch_size)
        prediction = probability.argmax(axis=1)
        fold_result = {
            "Prefix": args.prefix,
            "Method": "DiseaseCapsule",
            "Fold": fold,
            "N_train": int(len(train_idx)),
            "N_test": int(len(test_idx)),
            **metrics(y[test_idx], prediction),
        }
        fold_metrics.append(fold_result)
        selections.append({"Fold": fold, "BestParams": json.dumps(config, sort_keys=True)})
        fold_predictions.append(
            pd.DataFrame(
                {
                    "Prefix": args.prefix,
                    "Method": "DiseaseCapsule",
                    "Fold": fold,
                    "SampleIndex": test_idx,
                    "SampleID": sample_ids[test_idx],
                    "Y_true": y[test_idx],
                    "Y_pred": prediction,
                    "Prob_0": probability[:, 0],
                    "Prob_1": probability[:, 1],
                    "BestParams": json.dumps(config, sort_keys=True),
                }
            )
        )
        if args.save_models:
            torch.save(model.state_dict(), args.output_dir / f"{args.prefix}_fold_{fold}.pt")
        print(
            f"fold={fold} F1={fold_result['F1']:.4f} ACC={fold_result['Accuracy']:.4f} "
            f"params={json.dumps(config, sort_keys=True)}",
            flush=True,
        )

    metrics_frame = pd.DataFrame(fold_metrics)
    metrics_frame.to_csv(args.output_dir / f"{args.prefix}_fold_metrics.csv", index=False)
    pd.concat(fold_predictions, ignore_index=True).to_csv(
        args.output_dir / f"{args.prefix}_outer_fold_predictions.csv", index=False
    )
    pd.DataFrame(selections).to_csv(args.output_dir / f"{args.prefix}_selections.csv", index=False)
    summary = metrics_frame[["Precision", "Recall", "F1", "Accuracy"]].agg(["mean", "std"]).T
    summary.to_csv(args.output_dir / f"{args.prefix}_summary.csv")


if __name__ == "__main__":
    main()
