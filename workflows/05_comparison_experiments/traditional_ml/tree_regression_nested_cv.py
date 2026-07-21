import argparse
import csv
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_feature_cache_paths(feature_file: str) -> Tuple[str, str, str]:
    return (
        f"{feature_file}.float32.npy",
        f"{feature_file}.ids.txt",
        f"{feature_file}.meta.json",
    )


def load_feature_matrix_cached(
    feature_file: str,
    create_cache: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Load a sample-by-feature TSV, reusing a valid float32 cache when present."""
    cache_npy, cache_ids, cache_meta = get_feature_cache_paths(feature_file)
    st = os.stat(feature_file)
    expected = {"src_size": int(st.st_size), "src_mtime_ns": int(st.st_mtime_ns)}

    use_cache = False
    if os.path.exists(cache_npy) and os.path.exists(cache_ids) and os.path.exists(cache_meta):
        try:
            with open(cache_meta, "r") as f:
                meta = json.load(f)
            use_cache = (
                int(meta.get("src_size", -1)) == expected["src_size"]
                and int(meta.get("src_mtime_ns", -1)) == expected["src_mtime_ns"]
            )
        except Exception:
            use_cache = False

    if use_cache:
        x = np.load(cache_npy, mmap_mode="r")
        with open(cache_ids, "r") as f:
            ids = [line.rstrip("\n") for line in f]
        if x.ndim != 2 or x.shape[0] != len(ids):
            raise ValueError(
                f"Invalid feature cache: matrix shape={x.shape}, ids={len(ids)}"
            )
        print(f"[Cache] feature matrix loaded from {cache_npy} shape={x.shape}")
        return x, ids

    print(f"[Data] cache missing or stale; reading source table {feature_file}")
    frame = pd.read_csv(feature_file, sep="\t", index_col=0)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"Feature table is empty: {feature_file}")
    ids = normalize_ids(frame.index)
    # PLINK-derived tables commonly store the same sample identifier in both
    # FID and IID. Keep the first column as the index and discard only an IID
    # column that exactly duplicates it; never drop a genuine numeric feature.
    if str(frame.columns[0]).strip().lower() == "iid":
        iid_values = normalize_ids(frame.iloc[:, 0])
        if iid_values == ids:
            frame = frame.iloc[:, 1:]
    if frame.shape[1] == 0:
        raise ValueError(f"Feature table has no numeric feature columns: {feature_file}")
    if any(not sid for sid in ids):
        raise ValueError(f"Feature table contains an empty sample ID: {feature_file}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Feature table contains duplicated sample IDs: {feature_file}")
    try:
        x = frame.to_numpy(dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature table contains non-numeric values: {feature_file}") from exc

    if create_cache:
        np.save(cache_npy, x)
        with open(cache_ids, "w") as f:
            for sid in ids:
                f.write(f"{sid}\n")
        with open(cache_meta, "w") as f:
            json.dump({**expected, "shape": list(x.shape), "dtype": "float32"}, f, indent=2)
        print(f"[Cache] wrote {cache_npy} shape={x.shape}")

    return x, ids


def normalize_ids(values: Any) -> List[str]:
    return [str(v).lstrip("\ufeff").strip() for v in values]


def read_ids(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fold ID file not found: {path}")
    with open(path, "r", newline="") as f:
        return [sid for sid in normalize_ids(line.rstrip("\n") for line in f) if sid]


def read_phenotype(path: str) -> Tuple[str, List[str], np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phenotype file not found: {path}")

    ids = []
    values = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError(f"Phenotype file must contain ID and trait columns: {path}")
        trait = header[1]

        for line_number, row in enumerate(reader, start=2):
            if len(row) < 2:
                continue
            sid = normalize_ids([row[0]])[0]
            if not sid:
                continue
            try:
                value = float(row[1])
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric phenotype value at {path}:{line_number}: {row[1]!r}"
                ) from exc
            ids.append(sid)
            values.append(value)

    if not ids:
        raise ValueError(f"No phenotype records found: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Phenotype file contains duplicated IDs: {path}")
    return trait, ids, np.asarray(values, dtype=np.float32)


def write_csv_rows(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_csv(
    path: str,
    ids: List[str],
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []
    for sid, real, pred in zip(ids, y_true, y_pred):
        rows.append(
            {
                "ID": sid,
                "Fold": fold,
                "y_true": float(real),
                "y_pred": float(pred),
            }
        )
    write_csv_rows(path, ["ID", "Fold", "y_true", "y_pred"], rows)
    return rows


def sample_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "pearson": safe_pearson(y_true, y_pred),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def impute_from_training(
    train: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median-impute two matrices using only the supplied training rows."""
    train = np.asarray(train, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    medians = np.nan_to_num(medians, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    train_out = np.where(np.isfinite(train), train, medians)
    target_out = np.where(np.isfinite(target), target, medians)
    return train_out.astype(np.float32, copy=False), target_out.astype(np.float32, copy=False)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def rf_params_from_trial(trial: optuna.Trial, args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [200, 400, 600]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 16, 32]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 3, 5, 10]),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.05, 0.1]),
        "bootstrap": True,
        "random_state": seed,
        "n_jobs": args.n_jobs,
    }
    return params


def make_rf(params: Dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(**params)


def xgb_params_from_trial(trial: optuna.Trial, args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    try:
        from xgboost import XGBRegressor  # noqa: F401
    except ImportError as exc:
        raise ImportError("xgboost is required for --model_type xgboost") from exc

    params = {
        "objective": "reg:squarederror",
        "tree_method": args.xgb_tree_method,
        "device": args.xgb_device,
        "n_estimators": trial.suggest_categorical("n_estimators", [300, 600, 900]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.1]),
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7]),
        "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 5, 10]),
        "subsample": trial.suggest_categorical("subsample", [0.7, 0.9, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.2, 0.5, 0.8]),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "max_bin": args.xgb_max_bin,
        "random_state": seed,
        "n_jobs": args.n_jobs,
        "verbosity": 0,
    }
    return params


def make_xgboost(params: Dict[str, Any]) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**params)


def make_model(model_type: str, params: Dict[str, Any]) -> Any:
    if model_type == "rf":
        return make_rf(params)
    if model_type == "xgboost":
        return make_xgboost(params)
    raise ValueError(f"Unsupported model_type: {model_type}")


def suggest_params(
    trial: optuna.Trial,
    model_type: str,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, Any]:
    if model_type == "rf":
        return rf_params_from_trial(trial, args, seed)
    if model_type == "xgboost":
        return xgb_params_from_trial(trial, args, seed)
    raise ValueError(f"Unsupported model_type: {model_type}")


def objective(
    trial: optuna.Trial,
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    model_type: str,
    args: argparse.Namespace,
    fold: int,
) -> float:
    params = suggest_params(trial, model_type, args, seed=args.seed + fold * 1000 + trial.number)
    kf_inner = KFold(n_splits=args.inner_k, shuffle=True, random_state=args.seed + fold)
    scores = []

    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(kf_inner.split(x_train_all), start=1):
        model_params = dict(params)
        model_params["random_state"] = args.seed + fold * 1000 + inner_fold * 100 + trial.number
        model = make_model(model_type, model_params)
        x_inner_train, x_inner_val = impute_from_training(
            x_train_all[inner_train_idx],
            x_train_all[inner_val_idx],
        )
        model.fit(x_inner_train, y_train_all[inner_train_idx])
        preds = model.predict(x_inner_val)
        scores.append(r2_score(y_train_all[inner_val_idx], preds))

    score = float(np.mean(scores))
    if not np.isfinite(score):
        return -1e9
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested CV for tree-based GS models: RF and XGBoost")
    parser.add_argument("--feature_file", type=str, required=True)
    parser.add_argument("--y_file", type=str, required=True)
    parser.add_argument("--id_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./treegs_results")
    parser.add_argument("--model_type", type=str, choices=["rf", "xgboost", "xgb"], required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--outer_k", type=int, default=5)
    parser.add_argument("--inner_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--xgb_tree_method", type=str, default="hist")
    parser.add_argument("--xgb_device", type=str, default="cpu")
    parser.add_argument("--xgb_max_bin", type=int, default=64)
    parser.add_argument(
        "--create_cache",
        action="store_true",
        help="Write reusable .float32.npy/.ids.txt/.meta.json files beside the feature table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_type == "xgb":
        args.model_type = "xgboost"
    if args.inner_k < 2:
        raise ValueError("--inner_k must be >= 2 for nested CV")
    if args.n_trials < 1:
        raise ValueError("--n_trials must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model_label = "RF" if args.model_type == "rf" else "XGBoost"
    file_prefix = "rf" if args.model_type == "rf" else "xgboost"

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(json_ready(vars(args)), f, indent=2)

    x_all, x_ids = load_feature_matrix_cached(args.feature_file, create_cache=args.create_cache)
    x_ids = normalize_ids(x_ids)
    x_id_to_pos = {sid: i for i, sid in enumerate(x_ids)}

    trait, y_ids, y_values = read_phenotype(args.y_file)
    y_id_to_value = dict(zip(y_ids, y_values))
    y_id_set = set(y_ids)

    common_ids = [sid for sid in x_ids if sid in y_id_set]
    if not common_ids:
        raise ValueError("No overlapping sample IDs between feature and phenotype files")
    common_pos = np.asarray([x_id_to_pos[sid] for sid in common_ids], dtype=np.int64)
    x_common = np.asarray(x_all[common_pos], dtype=np.float32)
    y_common = np.asarray([y_id_to_value[sid] for sid in common_ids], dtype=np.float32)
    common_id_to_pos = {sid: i for i, sid in enumerate(common_ids)}

    print(
        f"[Data] matched IDs={len(common_ids)} features={x_common.shape[1]} "
        f"trait={trait} model={model_label}"
    )

    all_fold_metrics = []
    all_fold_predictions: List[Dict[str, Any]] = []
    all_best_params = []

    for fold in range(1, args.outer_k + 1):
        print(f"\n>>> {model_label} Outer Fold {fold}/{args.outer_k} <<<")
        train_ids = read_ids(os.path.join(args.id_dir, f"outer_fold_{fold}_train_IDs.txt"))
        test_ids = read_ids(os.path.join(args.id_dir, f"outer_fold_{fold}_test_IDs.txt"))

        missing_train = [sid for sid in train_ids if sid not in common_id_to_pos]
        missing_test = [sid for sid in test_ids if sid not in common_id_to_pos]
        if missing_train or missing_test:
            raise KeyError(
                f"Fold {fold} has IDs missing from aligned data: "
                f"train={len(missing_train)}, test={len(missing_test)}"
            )

        train_pos = np.asarray([common_id_to_pos[sid] for sid in train_ids], dtype=np.int64)
        test_pos = np.asarray([common_id_to_pos[sid] for sid in test_ids], dtype=np.int64)
        x_train = x_common[train_pos]
        x_test = x_common[test_pos]
        y_train = y_common[train_pos]
        y_test = y_common[test_pos]

        print(
            f"  Inner CV: {args.inner_k} folds x {args.n_trials} trials, "
            f"n_jobs={args.n_jobs}"
        )
        sampler = optuna.samplers.TPESampler(seed=args.seed + fold)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: objective(trial, x_train, y_train, args.model_type, args, fold),
            n_trials=args.n_trials,
            show_progress_bar=False,
        )

        best_params = suggest_params(
            study.best_trial,
            args.model_type,
            args,
            seed=args.seed + fold * 1000,
        )
        best_params["random_state"] = args.seed + fold * 1000
        best_record = {
            "Fold": fold,
            "best_inner_R2": float(study.best_value),
            "best_params": json_ready(best_params),
        }
        all_best_params.append(best_record)
        with open(os.path.join(args.output_dir, f"outer_fold_{fold}_best_params.json"), "w") as f:
            json.dump(best_record, f, indent=2)

        print(f"  Best inner R2={study.best_value:.4f}; fitting final outer model...")
        x_train, x_test = impute_from_training(x_train, x_test)
        final_model = make_model(args.model_type, best_params)
        final_model.fit(x_train, y_train)
        preds = final_model.predict(x_test)
        metrics = regression_metrics(y_test, preds)

        fold_metrics: Dict[str, Any] = {
            "Fold": fold,
            "pearson": metrics["pearson"],
            "MSE": metrics["MSE"],
            "R2": metrics["R2"],
        }
        all_fold_metrics.append(fold_metrics)

        fold_prediction_rows = write_prediction_csv(
            os.path.join(args.output_dir, f"{file_prefix}_outer_fold_{fold}_predictions.csv"),
            test_ids,
            fold,
            y_test,
            preds,
        )
        all_fold_predictions.extend(fold_prediction_rows)
        print(f"  Fold {fold}: R2={metrics['R2']:.4f}, PCC={metrics['pearson']:.4f}")

    write_csv_rows(
        os.path.join(args.output_dir, f"{file_prefix}_5fold_details.csv"),
        ["Fold", "pearson", "MSE", "R2"],
        all_fold_metrics,
    )
    write_csv_rows(
        os.path.join(args.output_dir, f"{file_prefix}_predictions.csv"),
        ["ID", "Fold", "y_true", "y_pred"],
        all_fold_predictions,
    )
    with open(os.path.join(args.output_dir, f"{file_prefix}_best_params.json"), "w") as f:
        json.dump(json_ready(all_best_params), f, indent=2)

    metric_arrays = {
        metric: np.asarray([row[metric] for row in all_fold_metrics], dtype=np.float64)
        for metric in ["pearson", "MSE", "R2"]
    }
    summary = {
        metric: {
            "mean": float(np.nanmean(values)),
            "std": sample_std(values),
        }
        for metric, values in metric_arrays.items()
    }

    with open(os.path.join(args.output_dir, f"{file_prefix}_nested_summary.txt"), "w") as f:
        f.write("Method\tMetric\tMean\tSD\n")
        for metric in ["pearson", "MSE", "R2"]:
            f.write(
                f"{model_label}\t{metric}\t"
                f"{summary[metric]['mean']:.4f}\t{summary[metric]['std']:.4f}\n"
            )

    print("\nSummary:")
    for metric in ["pearson", "MSE", "R2"]:
        print(f"  {metric}: mean={summary[metric]['mean']:.4f}, sd={summary[metric]['std']:.4f}")


if __name__ == "__main__":
    main()
