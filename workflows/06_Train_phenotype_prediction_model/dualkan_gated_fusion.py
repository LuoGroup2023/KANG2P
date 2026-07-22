#!/usr/bin/env python3
"""Dual-omics KAN runner for crop genome-wide prediction.

This version is designed for the ``data/plant/{Maize1404,Rice1495,Rice18K}``
layout. It keeps the evaluation honest: all feature
selection and scaling are fitted inside each training split only, then applied
to the held-out fold.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import kan as kan_layers  # noqa: E402


try:
    import optuna  # type: ignore

    HAS_OPTUNA = True
except Exception:
    optuna = None
    HAS_OPTUNA = False


warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch\\.cuda\\.amp.*")


DEFAULT_TRAITS: Dict[str, List[Tuple[str, str]]] = {
    "Maize1404": [
        ("PH", "PH_normal.tsv"),
        ("DTA", "DTA_normal.tsv"),
        ("KWPE", "KWPE_normal.tsv"),
        ("KNPE", "KNPE_normal.tsv"),
    ],
    "Rice1495": [
        ("Grain_width", "HZ_Grain_width_normal.txt"),
        ("Heading_date", "HZ_Heading_date_normal.txt"),
        ("Seed_setting_rate", "HZ_Seed_setting_rate_normal.txt"),
        ("Yield_per_plant", "HZ_Yield_per_plant_normal.txt"),
    ],
    "Rice18K": [
        ("Plant_height", "Plant_height.txt"),
        ("Culm_length", "Culm_length.txt"),
        ("Grain_length", "Grain_length.txt"),
        ("Grain_width", "Grain_width.txt"),
        ("Grain_yield", "Grain_yield.txt"),
    ],
}

OMICS_FILES: Dict[str, Tuple[str, str]] = {
    "Maize1404": ("X.txt", "Exp.txt"),
    "Rice1495": ("X.txt", "PE.txt"),
    "Rice18K": ("X.txt", "PE.txt"),
}


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def safe_name(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)).strip("_")


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if y_true.size < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if y_true.size < 3 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(spearmanr(y_true, y_pred)[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "PCC": safe_pearson(y_true, y_pred),
        "SCC": safe_spearman(y_true, y_pred),
    }


def score_for_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    m = regression_metrics(y_true, y_pred)
    if metric == "mse":
        return -m["MSE"]
    if metric == "composite":
        return 0.30 * m["R2"] + 0.35 * m["PCC"] + 0.35 * m["SCC"]
    return m[metric.upper()]


def read_ids(path: Path) -> List[str]:
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def read_trait(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] < 2:
        raise ValueError(f"Trait file needs at least two columns: {path}")
    ids = df.iloc[:, 0].astype(str)
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    out = pd.Series(y.to_numpy(dtype=np.float32), index=ids, name=df.columns[1])
    return out.dropna()


class FeatureBundle:
    """Feature table backed by float32 memmap cache when available."""

    def __init__(self, table_path: Path, create_cache: bool = False) -> None:
        self.table_path = table_path
        npy_path = Path(str(table_path) + ".float32.npy")
        id_path = Path(str(table_path) + ".ids.txt")
        if npy_path.exists() and id_path.exists():
            self.values = np.load(npy_path, mmap_mode="r")
            self.ids = read_ids(id_path)
            self.source = str(npy_path)
        else:
            df = pd.read_csv(table_path, sep="\t", index_col=0)
            df.index = df.index.astype(str)
            arr = df.values.astype(np.float32, copy=False)
            self.values = arr
            self.ids = df.index.tolist()
            self.source = str(table_path)
            if create_cache:
                np.save(npy_path, arr)
                pd.Series(self.ids).to_csv(id_path, index=False, header=False)
        if len(self.ids) != self.values.shape[0]:
            raise ValueError(f"{table_path}: ids length {len(self.ids)} != rows {self.values.shape[0]}")
        self.id_to_row = {sid: i for i, sid in enumerate(self.ids)}


@dataclass
class DatasetBundle:
    dataset: str
    data_dir: Path
    geno: FeatureBundle
    expr: FeatureBundle


@dataclass
class Scaler1D:
    mean: float
    std: float

    @classmethod
    def fit(cls, y: np.ndarray) -> "Scaler1D":
        mean = float(np.mean(y))
        std = float(np.std(y))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return (y * self.std + self.mean).astype(np.float32)


@dataclass
class FoldArrays:
    xg_train: np.ndarray
    xe_train: np.ndarray
    y_train: np.ndarray
    xg_val: np.ndarray
    xe_val: np.ndarray
    y_val: np.ndarray
    y_val_raw: np.ndarray
    y_scaler: Scaler1D
    geno_cols: np.ndarray
    expr_cols: np.ndarray


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + x)


class BranchEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, add_base: bool = True) -> None:
        super().__init__()
        scale = 1.0 / math.sqrt(max(1, in_features * grid_size))
        self.coeff = nn.Parameter(torch.randn(out_features, in_features, 2 * grid_size) * scale)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.base = nn.Linear(in_features, out_features) if add_base else None
        self.register_buffer("freqs", torch.arange(1, grid_size + 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(x).unsqueeze(-1) * self.freqs.view(1, 1, -1)
        basis = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        out = torch.einsum("bik,oik->bo", basis, self.coeff) + self.bias
        if self.base is not None:
            out = out + self.base(x)
        return out


class DualOmicsRegressor(nn.Module):
    def __init__(
        self,
        geno_dim: int,
        expr_dim: int,
        hidden_dim: int,
        latent_dim: int,
        head_dim: int,
        dropout: float,
        grid_size: int,
        head_type: str,
    ) -> None:
        super().__init__()
        self.encoder_g = BranchEncoder(geno_dim, hidden_dim, latent_dim, dropout)
        self.encoder_e = BranchEncoder(expr_dim, hidden_dim, latent_dim, dropout)
        self.decoder_g = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, geno_dim),
        )
        self.decoder_e = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, expr_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Dropout(dropout),
            nn.Linear(head_dim, latent_dim * 2),
            nn.Sigmoid(),
        )
        pred_dim = latent_dim * 5
        if head_type == "fourier":
            self.pred = nn.Sequential(
                FourierKANLayer(pred_dim, head_dim, grid_size=grid_size),
                nn.GELU(),
                nn.LayerNorm(head_dim),
                nn.Dropout(dropout),
                FourierKANLayer(head_dim, 1, grid_size=grid_size),
            )
        elif head_type == "spline":
            self.pred = nn.Sequential(
                kan_layers.KANLinear(pred_dim, head_dim, grid_size=grid_size, spline_order=3),
                nn.GELU(),
                nn.LayerNorm(head_dim),
                nn.Dropout(dropout),
                kan_layers.KANLinear(head_dim, 1, grid_size=grid_size, spline_order=3),
            )
        else:
            raise ValueError(f"Unknown head_type={head_type!r}; expected 'spline' or 'fourier'.")

    def forward(self, xg: torch.Tensor, xe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zg = self.encoder_g(xg)
        ze = self.encoder_e(xe)
        gates = self.fusion_gate(torch.cat([zg, ze], dim=1))
        gg, ge = torch.chunk(gates, 2, dim=1)
        zg2 = gg * zg
        ze2 = ge * ze
        fused = torch.cat([zg2, ze2, zg2 * ze2, torch.abs(zg2 - ze2), zg2 + ze2], dim=1)
        pred = self.pred(fused).squeeze(-1)
        return pred, self.decoder_g(zg), self.decoder_e(ze)


def parse_folds(text: str, n_outer_folds: int) -> List[int]:
    if text.strip().lower() in {"", "all"}:
        return list(range(1, n_outer_folds + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def resolve_traits(args: argparse.Namespace, dataset: str) -> List[Tuple[str, Path]]:
    data_dir = Path(args.data_root) / dataset
    defaults = [(name, data_dir / rel) for name, rel in DEFAULT_TRAITS[dataset]]
    if not args.trait:
        return defaults
    selected = []
    default_by_name = {name: path for name, path in defaults}
    for item in args.trait:
        if ":" in item:
            ds, rest = item.split(":", 1)
            if ds != dataset:
                continue
            item = rest
        if "=" in item:
            name, path_text = item.split("=", 1)
            path = Path(path_text)
            if not path.is_absolute():
                path = data_dir / path
            selected.append((name, path))
        elif item in default_by_name:
            selected.append((item, default_by_name[item]))
    return selected


def select_topk_by_corr(
    values: np.ndarray,
    train_rows: np.ndarray,
    y_train: np.ndarray,
    k: int,
    chunk_cols: int,
) -> np.ndarray:
    n_features = values.shape[1]
    if k <= 0 or k >= n_features:
        return np.arange(n_features, dtype=np.int64)

    rows = np.asarray(train_rows, dtype=np.int64)
    y = y_train.astype(np.float64, copy=False)
    y = y - np.mean(y)
    y_norm = math.sqrt(float(np.dot(y, y))) + 1e-12
    scores = np.empty(n_features, dtype=np.float32)

    for start in range(0, n_features, chunk_cols):
        end = min(start + chunk_cols, n_features)
        cols = np.arange(start, end, dtype=np.int64)
        block = np.asarray(values[np.ix_(rows, cols)], dtype=np.float32)
        block = block.astype(np.float64, copy=False)
        block -= block.mean(axis=0, keepdims=True)
        denom = np.sqrt(np.sum(block * block, axis=0)) * y_norm + 1e-12
        corr = np.abs((block.T @ y) / denom)
        scores[start:end] = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return top.astype(np.int64)


def materialize_scaled(
    values: np.ndarray,
    train_rows: np.ndarray,
    target_rows: np.ndarray,
    cols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    train = np.asarray(values[np.ix_(train_rows, cols)], dtype=np.float32)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    train = (train - mean) / std
    target = np.asarray(values[np.ix_(target_rows, cols)], dtype=np.float32)
    target = (target - mean) / std
    return train.astype(np.float32, copy=False), target.astype(np.float32, copy=False)


def make_loader(
    xg: np.ndarray,
    xe: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(xg, dtype=torch.float32),
        torch.tensor(xe, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def parse_gpu_ids(text: str) -> List[int]:
    if text.lower() == "all":
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in text.split(",") if x.strip()]


def resolve_device(args: argparse.Namespace) -> Tuple[torch.device, List[int], bool]:
    if args.device == "auto":
        device_text = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_text = args.device
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA unavailable in this process; falling back to CPU.")
        return torch.device("cpu"), [], False
    device = torch.device(device_text)
    gpu_ids = parse_gpu_ids(args.gpu_ids) if device.type == "cuda" else []
    if device.type == "cuda" and not gpu_ids:
        gpu_ids = [device.index or 0]
    use_data_parallel = device.type == "cuda" and len(gpu_ids) > 1 and not args.disable_data_parallel
    if use_data_parallel:
        device = torch.device(f"cuda:{gpu_ids[0]}")
    return device, gpu_ids, use_data_parallel


def make_model(
    params: Dict[str, object],
    geno_dim: int,
    expr_dim: int,
    device: torch.device,
    gpu_ids: Sequence[int],
    use_data_parallel: bool,
) -> nn.Module:
    model = DualOmicsRegressor(
        geno_dim=geno_dim,
        expr_dim=expr_dim,
        hidden_dim=int(params["hidden_dim"]),
        latent_dim=int(params["latent_dim"]),
        head_dim=int(params["head_dim"]),
        dropout=float(params["dropout"]),
        grid_size=int(params["grid_size"]),
        head_type=str(params.get("head_type", "spline")),
    ).to(device)
    if use_data_parallel:
        model = nn.DataParallel(model, device_ids=list(gpu_ids), output_device=gpu_ids[0])
    return model


def model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    inner = model.module if isinstance(model, nn.DataParallel) else model
    return {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()}


def train_one_model(
    params: Dict[str, object],
    fold: FoldArrays,
    device: torch.device,
    gpu_ids: Sequence[int],
    use_data_parallel: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    use_amp: bool,
    opt_metric: str,
) -> nn.Module:
    seed_everything(seed)
    model = make_model(params, fold.xg_train.shape[1], fold.xe_train.shape[1], device, gpu_ids, use_data_parallel)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(params["epochs"])))
    mse_loss = nn.MSELoss()
    train_loader = make_loader(
        fold.xg_train,
        fold.xe_train,
        fold.y_train,
        int(params["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    xg_val = torch.tensor(fold.xg_val, dtype=torch.float32, device=device)
    xe_val = torch.tensor(fold.xe_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(fold.y_val, dtype=torch.float32, device=device)

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_score = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    alpha = float(params.get("alpha", 0.05))

    for _ in range(int(params["epochs"])):
        model.train()
        for bg, be, by in train_loader:
            bg = bg.to(device, non_blocking=pin_memory)
            be = be.to(device, non_blocking=pin_memory)
            by = by.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred, rec_g, rec_e = model(bg, be)
                loss_pred = mse_loss(pred, by)
                loss_rec = mse_loss(rec_g, bg) + mse_loss(rec_e, be)
                loss = loss_pred + alpha * loss_rec
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred_val, _, _ = model(xg_val, xe_val)
        pred_raw = fold.y_scaler.inverse_transform(pred_val.detach().float().cpu().numpy())
        score = score_for_metric(fold.y_val_raw, pred_raw, opt_metric)
        if score > best_score + 1e-7:
            best_score = score
            best_state = model_state_dict(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(params["patience"]):
                break

    if best_state is not None:
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.load_state_dict(best_state)

    del train_loader, xg_val, xe_val, y_val
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


def predict_model(
    model: nn.Module,
    xg: np.ndarray,
    xe: np.ndarray,
    y_scaler: Scaler1D,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    ds = TensorDataset(torch.tensor(xg, dtype=torch.float32), torch.tensor(xe, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds: List[np.ndarray] = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    model.eval()
    with torch.no_grad():
        for bg, be in loader:
            bg = bg.to(device)
            be = be.to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                p, _, _ = model(bg, be)
            preds.append(p.detach().float().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return y_scaler.inverse_transform(pred)


def sample_params_from_trial(trial: object, args: argparse.Namespace) -> Dict[str, object]:
    return {
        "lr": trial.suggest_categorical("lr", [1e-4, 3e-4, 8e-4]),  # type: ignore[attr-defined]
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),  # type: ignore[attr-defined]
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 384]),  # type: ignore[attr-defined]
        "latent_dim": trial.suggest_categorical("latent_dim", [32, 64, 96]),  # type: ignore[attr-defined]
        "head_dim": trial.suggest_categorical("head_dim", [64, 128, 192]),  # type: ignore[attr-defined]
        "dropout": trial.suggest_categorical("dropout", [0.05, 0.1, 0.2, 0.3]),  # type: ignore[attr-defined]
        "alpha": trial.suggest_categorical("alpha", [0.0, 0.02, 0.05, 0.1, 0.2]),  # type: ignore[attr-defined]
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4]),  # type: ignore[attr-defined]
        "grid_size": trial.suggest_categorical("grid_size", [3, 5]),  # type: ignore[attr-defined]
        "head_type": args.head_type,
        "epochs": args.max_epochs,
        "patience": args.patience,
    }


def sample_params_random(rng: random.Random, args: argparse.Namespace) -> Dict[str, object]:
    space = {
        "lr": [1e-4, 3e-4, 8e-4],
        "batch_size": [128, 256, 512],
        "hidden_dim": [128, 256, 384],
        "latent_dim": [32, 64, 96],
        "head_dim": [64, 128, 192],
        "dropout": [0.05, 0.1, 0.2, 0.3],
        "alpha": [0.0, 0.02, 0.05, 0.1, 0.2],
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
        "grid_size": [3, 5],
    }
    params = {k: rng.choice(v) for k, v in space.items()}
    params["head_type"] = args.head_type
    params["epochs"] = args.max_epochs
    params["patience"] = args.patience
    return params


def prepare_fold_arrays(
    bundle: DatasetBundle,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    y: pd.Series,
    args: argparse.Namespace,
) -> FoldArrays:
    train_g = np.array([bundle.geno.id_to_row[i] for i in train_ids], dtype=np.int64)
    val_g = np.array([bundle.geno.id_to_row[i] for i in val_ids], dtype=np.int64)
    train_e = np.array([bundle.expr.id_to_row[i] for i in train_ids], dtype=np.int64)
    val_e = np.array([bundle.expr.id_to_row[i] for i in val_ids], dtype=np.int64)
    y_train_raw = y.loc[list(train_ids)].to_numpy(dtype=np.float32)
    y_val_raw = y.loc[list(val_ids)].to_numpy(dtype=np.float32)

    geno_cols = select_topk_by_corr(bundle.geno.values, train_g, y_train_raw, args.topk_geno, args.chunk_cols)
    expr_cols = select_topk_by_corr(bundle.expr.values, train_e, y_train_raw, args.topk_expr, args.chunk_cols)

    xg_train, xg_val = materialize_scaled(bundle.geno.values, train_g, val_g, geno_cols)
    xe_train, xe_val = materialize_scaled(bundle.expr.values, train_e, val_e, expr_cols)
    y_scaler = Scaler1D.fit(y_train_raw) if args.standardize_y else Scaler1D(0.0, 1.0)
    y_train = y_scaler.transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    return FoldArrays(xg_train, xe_train, y_train, xg_val, xe_val, y_val, y_val_raw, y_scaler, geno_cols, expr_cols)


def tune_params(
    inner_folds: Sequence[FoldArrays],
    args: argparse.Namespace,
    device: torch.device,
    gpu_ids: Sequence[int],
    use_data_parallel: bool,
    pin_memory: bool,
    use_amp: bool,
    seed: int,
    out_dir: Path,
) -> Tuple[Dict[str, object], float]:
    history: List[Dict[str, object]] = []

    def eval_params(params: Dict[str, object], trial_index: int) -> float:
        scores = []
        for fold_i, fold in enumerate(inner_folds, start=1):
            model = train_one_model(
                params,
                fold,
                device=device,
                gpu_ids=gpu_ids,
                use_data_parallel=use_data_parallel,
                seed=seed + 1000 * trial_index + fold_i,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                use_amp=use_amp,
                opt_metric=args.opt_metric,
            )
            pred = predict_model(model, fold.xg_val, fold.xe_val, fold.y_scaler, device, int(params["batch_size"]), use_amp)
            scores.append(score_for_metric(fold.y_val_raw, pred, args.opt_metric))
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return float(np.mean(scores))

    if HAS_OPTUNA and not args.force_random_search:
        study = optuna.create_study(  # type: ignore[union-attr]
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),  # type: ignore[union-attr]
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),  # type: ignore[union-attr]
        )

        def objective(trial: object) -> float:
            params = sample_params_from_trial(trial, args)
            value = eval_params(params, int(trial.number))  # type: ignore[attr-defined]
            history.append({"trial": int(trial.number), "score": value, **params})  # type: ignore[attr-defined]
            return value

        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        best_score = float(study.best_value)
        best_params.update({"epochs": args.max_epochs, "patience": args.patience, "head_type": args.head_type})
    else:
        if not HAS_OPTUNA:
            print("[Info] optuna is not installed in this environment; using seeded random search.")
        rng = random.Random(seed)
        best_score = -float("inf")
        best_params: Dict[str, object] = {}
        for trial_i in range(args.n_trials):
            params = sample_params_random(rng, args)
            value = eval_params(params, trial_i)
            history.append({"trial": trial_i, "score": value, **params})
            pd.DataFrame(history).to_csv(out_dir / "tuning_history.csv", index=False)
            print(
                f"[tune] trial {trial_i + 1}/{args.n_trials}: "
                f"{args.opt_metric}={value:.6f}",
                flush=True,
            )
            if value > best_score:
                best_score = value
                best_params = params

    pd.DataFrame(history).to_csv(out_dir / "tuning_history.csv", index=False)
    return best_params, best_score


def run_dataset_trait(
    bundle: DatasetBundle,
    trait_name: str,
    trait_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    gpu_ids: Sequence[int],
    use_data_parallel: bool,
    pin_memory: bool,
    use_amp: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = read_trait(trait_path)
    common_ids = [
        sid
        for sid in bundle.geno.ids
        if sid in bundle.expr.id_to_row and sid in y.index
    ]
    if not common_ids:
        raise RuntimeError(f"{bundle.dataset}/{trait_name}: no shared sample IDs.")
    y = y.loc[common_ids]

    folds = parse_folds(args.folds, args.n_outer_folds)
    trait_dir = Path(args.output_dir) / bundle.dataset / safe_name(trait_name)
    trait_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n=== {bundle.dataset} / {trait_name}: n={len(common_ids)}, "
        f"geno_dim={bundle.geno.values.shape[1]}, expr_dim={bundle.expr.values.shape[1]} ==="
    )

    fold_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []
    id_set = set(common_ids)

    for outer_fold in folds:
        train_ids_raw = read_ids(bundle.data_dir / f"outer_fold_{outer_fold}_train_IDs.txt")
        test_ids_raw = read_ids(bundle.data_dir / f"outer_fold_{outer_fold}_test_IDs.txt")
        train_ids = [sid for sid in train_ids_raw if sid in id_set]
        test_ids = [sid for sid in test_ids_raw if sid in id_set]
        if len(train_ids) < args.inner_folds + 2 or len(test_ids) == 0:
            raise RuntimeError(f"{bundle.dataset}/{trait_name}/fold{outer_fold}: bad train/test sizes.")
        print(f"[{bundle.dataset}/{trait_name}] outer fold {outer_fold}: train={len(train_ids)} test={len(test_ids)}")

        kf = KFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + outer_fold)
        train_ids_arr = np.array(train_ids, dtype=object)
        inner_folds: List[FoldArrays] = []
        for tr_idx, val_idx in kf.split(train_ids_arr):
            inner_folds.append(
                prepare_fold_arrays(bundle, train_ids_arr[tr_idx].tolist(), train_ids_arr[val_idx].tolist(), y, args)
            )

        fold_dir = trait_dir / f"outer_fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_params, best_inner_score = tune_params(
            inner_folds,
            args,
            device,
            gpu_ids,
            use_data_parallel,
            pin_memory,
            use_amp,
            args.seed + 100 * outer_fold,
            fold_dir,
        )
        with open(fold_dir / "best_params.json", "w", encoding="utf-8") as fh:
            json.dump(best_params, fh, indent=2)
        print(f"[{bundle.dataset}/{trait_name}] fold {outer_fold} best inner {args.opt_metric}: {best_inner_score:.6f}")

        final_train, final_val = train_test_split(
            train_ids,
            test_size=args.val_fraction,
            random_state=args.seed + 1000 + outer_fold,
            shuffle=True,
        )
        final_fold = prepare_fold_arrays(bundle, final_train, final_val, y, args)

        test_g_rows = np.array([bundle.geno.id_to_row[i] for i in test_ids], dtype=np.int64)
        test_e_rows = np.array([bundle.expr.id_to_row[i] for i in test_ids], dtype=np.int64)
        full_train_g_rows = np.array([bundle.geno.id_to_row[i] for i in final_train], dtype=np.int64)
        full_train_e_rows = np.array([bundle.expr.id_to_row[i] for i in final_train], dtype=np.int64)
        _, xg_test = materialize_scaled(bundle.geno.values, full_train_g_rows, test_g_rows, final_fold.geno_cols)
        _, xe_test = materialize_scaled(bundle.expr.values, full_train_e_rows, test_e_rows, final_fold.expr_cols)
        y_test = y.loc[test_ids].to_numpy(dtype=np.float32)

        pred_ens = []
        for ens_i in range(args.ensemble_size):
            model = train_one_model(
                best_params,
                final_fold,
                device=device,
                gpu_ids=gpu_ids,
                use_data_parallel=use_data_parallel,
                seed=args.seed + 10000 + 97 * ens_i + outer_fold,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                use_amp=use_amp,
                opt_metric=args.opt_metric,
            )
            pred_ens.append(
                predict_model(model, xg_test, xe_test, final_fold.y_scaler, device, int(best_params["batch_size"]), use_amp)
            )
            if ens_i == 0:
                torch.save(model_state_dict(model), fold_dir / "best_model_ensemble0.pt")
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        pred_test = np.mean(np.stack(pred_ens, axis=0), axis=0)
        metrics = regression_metrics(y_test, pred_test)
        print(
            f"[{bundle.dataset}/{trait_name}] fold {outer_fold} test: "
            f"R2={metrics['R2']:.4f} MSE={metrics['MSE']:.6f} "
            f"PCC={metrics['PCC']:.4f} SCC={metrics['SCC']:.4f}"
        )
        fold_rows.append(
            {
                "Dataset": bundle.dataset,
                "Trait": trait_name,
                "OuterFold": outer_fold,
                "N_Train": len(train_ids),
                "N_Test": len(test_ids),
                "Inner_Best_Score": best_inner_score,
                "R2": metrics["R2"],
                "MSE": metrics["MSE"],
                "PCC": metrics["PCC"],
                "SCC": metrics["SCC"],
                "TopK_Geno": int(len(final_fold.geno_cols)),
                "TopK_Expr": int(len(final_fold.expr_cols)),
                "EnsembleSize": args.ensemble_size,
                "HeadType": args.head_type,
                "BestParams": json.dumps(best_params, sort_keys=True),
            }
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "Dataset": bundle.dataset,
                    "Trait": trait_name,
                    "OuterFold": outer_fold,
                    "ID": test_ids,
                    "y_true": y_test,
                    "y_pred": pred_test,
                }
            )
        )

    metrics_df = pd.DataFrame(fold_rows)
    preds_df = pd.concat(pred_frames, axis=0, ignore_index=True)
    metrics_df.to_csv(trait_dir / "dualkan_fold_metrics.csv", index=False)
    preds_df.to_csv(trait_dir / "dualkan_predictions.csv", index=False)

    summary = {
        "Dataset": bundle.dataset,
        "Trait": trait_name,
        "R2_mean": float(metrics_df["R2"].mean()),
        "R2_std": float(metrics_df["R2"].std(ddof=1)),
        "MSE_mean": float(metrics_df["MSE"].mean()),
        "MSE_std": float(metrics_df["MSE"].std(ddof=1)),
        "PCC_mean": float(metrics_df["PCC"].mean()),
        "PCC_std": float(metrics_df["PCC"].std(ddof=1)),
        "SCC_mean": float(metrics_df["SCC"].mean()),
        "SCC_std": float(metrics_df["SCC"].std(ddof=1)),
    }
    with open(trait_dir / "dualkan_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return metrics_df, preds_df


def load_dataset(dataset: str, args: argparse.Namespace) -> DatasetBundle:
    data_dir = Path(args.data_root) / dataset
    geno_rel, expr_rel = OMICS_FILES[dataset]
    geno = FeatureBundle(data_dir / geno_rel, create_cache=args.create_cache)
    expr = FeatureBundle(data_dir / expr_rel, create_cache=args.create_cache)
    return DatasetBundle(dataset=dataset, data_dir=data_dir, geno=geno, expr=expr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualKAN dual-omics crop genome-wide prediction")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data_root", type=str, default=str(repo_root / "data" / "plant"))
    parser.add_argument("--datasets", nargs="+", default=["Maize1404", "Rice1495", "Rice18K"])
    parser.add_argument("--trait", action="append", help="Trait filter, e.g. PH or Maize1404:PH or PH=file.tsv")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(repo_root / "outputs" / "dualkan_gated_fusion"),
    )
    parser.add_argument("--n_outer_folds", type=int, default=5)
    parser.add_argument("--folds", type=str, default="all")
    parser.add_argument("--inner_folds", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=12)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1521024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu_ids", type=str, default="all")
    parser.add_argument("--disable_data_parallel", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=1, choices=[0, 1])
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--force_random_search", action="store_true")
    parser.add_argument("--opt_metric", type=str, default="composite", choices=["composite", "pcc", "scc", "r2", "mse"])
    parser.add_argument("--head_type", type=str, default="spline", choices=["spline", "fourier"])
    parser.add_argument("--topk_geno", type=int, default=4096)
    parser.add_argument("--topk_expr", type=int, default=1024)
    parser.add_argument("--chunk_cols", type=int, default=2048)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--standardize_y", type=int, default=1, choices=[0, 1])
    parser.add_argument("--create_cache", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny sanity check: first dataset/trait/fold only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.datasets = args.datasets[:1]
        args.folds = "1"
        args.n_trials = 1
        args.inner_folds = 2
        args.ensemble_size = 1
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = 1
        args.topk_geno = min(args.topk_geno, 64)
        args.topk_expr = min(args.topk_expr, 32)

    seed_everything(args.seed, deterministic=args.deterministic)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device, gpu_ids, use_data_parallel = resolve_device(args)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    pin_memory = bool(args.pin_memory) and device.type == "cuda"
    use_amp = (not args.disable_amp) and device.type == "cuda"
    print(
        f"Runtime: device={device}, gpu_ids={gpu_ids}, "
        f"data_parallel={use_data_parallel}, amp={use_amp}, optuna={HAS_OPTUNA}"
    )

    config = vars(args).copy()
    config.update({"resolved_device": str(device), "gpu_ids": gpu_ids, "data_parallel": use_data_parallel})
    with open(Path(args.output_dir) / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    all_metrics = []
    all_predictions = []
    for dataset in args.datasets:
        if dataset not in DEFAULT_TRAITS:
            raise ValueError(f"Unknown dataset {dataset}; available: {sorted(DEFAULT_TRAITS)}")
        bundle = load_dataset(dataset, args)
        print(f"Loaded {dataset}: geno={bundle.geno.source}, expr={bundle.expr.source}")
        traits = resolve_traits(args, dataset)
        if args.smoke:
            traits = traits[:1]
        for trait_name, trait_path in traits:
            metrics_df, preds_df = run_dataset_trait(
                bundle,
                trait_name,
                trait_path,
                args,
                device,
                gpu_ids,
                use_data_parallel,
                pin_memory,
                use_amp,
            )
            all_metrics.append(metrics_df)
            all_predictions.append(preds_df)

    metrics = pd.concat(all_metrics, axis=0, ignore_index=True)
    predictions = pd.concat(all_predictions, axis=0, ignore_index=True)
    metrics.to_csv(Path(args.output_dir) / "dualkan_all_fold_metrics.csv", index=False)
    predictions.to_csv(Path(args.output_dir) / "dualkan_all_predictions.csv", index=False)

    summary = (
        metrics.groupby(["Dataset", "Trait"], as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            PCC_mean=("PCC", "mean"),
            PCC_std=("PCC", "std"),
            SCC_mean=("SCC", "mean"),
            SCC_std=("SCC", "std"),
        )
        .sort_values(["Dataset", "Trait"])
    )
    summary.to_csv(Path(args.output_dir) / "dualkan_all_summary.csv", index=False)
    print("\n=== DualKAN summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
