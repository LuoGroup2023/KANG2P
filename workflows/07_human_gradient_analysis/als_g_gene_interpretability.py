#!/usr/bin/env python
"""Gene-level interpretability for ALS genotype-only KAN/DualBranch-KAN inputs.

The ALS genotype feature table is already grouped as chr:GENE:component.  This
script aggregates multiple explainability signals from feature level to gene
level and also reports a clearly labelled wet-lab-prior-aware ranking.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier


SEED = 1521024


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1:
        return np.argmax(y, axis=1).astype(np.int64)
    return y.astype(np.int64)


def parse_gene_columns(columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for i, col in enumerate(columns):
        parts = str(col).split(":")
        if len(parts) >= 3:
            chrom, gene, component = parts[0], parts[1], ":".join(parts[2:])
        elif len(parts) == 2:
            chrom, gene, component = "", parts[0], parts[1]
        else:
            chrom, gene, component = "", str(col), ""
        rows.append({"feature_index": i, "feature": str(col), "chrom": chrom, "gene": gene, "component": component})
    return pd.DataFrame(rows)


def load_genotype_pkl(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    x_obj, y_obj = data[0], data[1]
    if not isinstance(x_obj, pd.DataFrame):
        raise TypeError("Expected genotype PKL first item to be a pandas DataFrame with gene-labelled columns.")
    y = normalize_labels(y_obj)
    if x_obj.shape[0] != len(y):
        raise ValueError(f"X/Y mismatch: {x_obj.shape[0]} rows vs {len(y)} labels")
    return x_obj, y


def load_pe_txt(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep="\t")
    if df.iloc[:, 0].dtype == object or df.iloc[:, 0].dtype == str:
        df = df.set_index(df.columns[0])
    return df.values.astype(np.float32, copy=False)


@dataclass
class GeneIndex:
    feature_meta: pd.DataFrame
    genes: List[str]
    gene_to_indices: Dict[str, np.ndarray]
    gene_chrom: Dict[str, str]

    @classmethod
    def from_columns(cls, columns: Sequence[str]) -> "GeneIndex":
        meta = parse_gene_columns(columns)
        genes = sorted(meta["gene"].unique().tolist())
        gene_to_indices = {
            gene: meta.index[meta["gene"].eq(gene)].to_numpy(dtype=np.int64)
            for gene in genes
        }
        gene_chrom = (
            meta.drop_duplicates("gene")
            .set_index("gene")["chrom"]
            .astype(str)
            .to_dict()
        )
        return cls(meta, genes, gene_to_indices, gene_chrom)


def aggregate_feature_scores(
    scores: np.ndarray,
    gene_index: GeneIndex,
    *,
    suffix: str,
    signed_scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    rows = []
    scores = np.asarray(scores, dtype=np.float64)
    signed_scores = np.asarray(signed_scores, dtype=np.float64) if signed_scores is not None else None
    for gene in gene_index.genes:
        idx = gene_index.gene_to_indices[gene]
        vals = scores[idx]
        signed = signed_scores[idx] if signed_scores is not None else vals
        max_pos = int(np.nanargmax(vals))
        rows.append(
            {
                "gene": gene,
                "chrom": gene_index.gene_chrom.get(gene, ""),
                "n_features": int(len(idx)),
                f"{suffix}_max": float(np.nanmax(vals)),
                f"{suffix}_mean": float(np.nanmean(vals)),
                f"{suffix}_rms": float(np.sqrt(np.nanmean(vals ** 2))),
                f"{suffix}_sum": float(np.nansum(vals)),
                f"{suffix}_signed_at_max": float(signed[max_pos]),
                f"{suffix}_top_feature": gene_index.feature_meta.iloc[int(idx[max_pos])]["feature"],
            }
        )
    return pd.DataFrame(rows)


def merge_gene_score_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0]
    for frame in frames[1:]:
        shared = [c for c in ("gene", "chrom", "n_features") if c in out.columns and c in frame.columns]
        use_cols = [c for c in frame.columns if c not in {"chrom", "n_features"} or c not in out.columns]
        out = out.merge(frame[use_cols], on="gene", how="outer")
        for col in ("chrom", "n_features"):
            if col not in out.columns and col in frame.columns:
                out[col] = frame[col]
    return out


def robust_scale_train_matrix(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    z = ((x - mean) / std).astype(np.float32, copy=False)
    return z, mean, std


def standardized_mean_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = x[y == 0]
    x1 = x[y == 1]
    m0 = x0.mean(axis=0, dtype=np.float64)
    m1 = x1.mean(axis=0, dtype=np.float64)
    v0 = x0.var(axis=0, dtype=np.float64)
    v1 = x1.var(axis=0, dtype=np.float64)
    pooled = np.sqrt(0.5 * (v0 + v1))
    pooled[~np.isfinite(pooled) | (pooled < 1e-8)] = 1.0
    return ((m1 - m0) / pooled).astype(np.float32)


def train_sgd_logistic_coefficients(x_z: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=3e-5,
        l1_ratio=0.05,
        max_iter=400,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(x_z, y)
    return clf.coef_.reshape(-1).astype(np.float32)


class FourierKANLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8, add_base: bool = True):
        super().__init__()
        scale = 1.0 / math.sqrt(max(1, in_features * grid_size))
        self.coeff = torch.nn.Parameter(torch.randn(out_features, in_features, 2 * grid_size) * scale)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.base = torch.nn.Linear(in_features, out_features) if add_base else None
        self.register_buffer("freqs", torch.arange(1, grid_size + 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(x).unsqueeze(-1) * self.freqs.view(1, 1, -1)
        basis = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        out = torch.einsum("bik,oik->bo", basis, self.coeff) + self.bias
        if self.base is not None:
            out = out + self.base(x)
        return out


class KANG2PNet(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 192,
        kan_hidden: int = 96,
        grid_size: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_dim, projection_dim),
            torch.nn.LayerNorm(projection_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.kan1 = FourierKANLayer(projection_dim, kan_hidden, grid_size=grid_size)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        self.kan2 = FourierKANLayer(kan_hidden, 2, grid_size=grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.project(x)
        z = self.drop(self.act(self.kan1(z)))
        return self.kan2(z)


class BranchEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, projection_dim),
            torch.nn.LayerNorm(projection_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionExpert(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.LayerNorm(output_dim),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoisyTopKGating(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        expert_hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.2,
        loss_coef: float = 0.05,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.loss_coef = loss_coef
        self.experts = torch.nn.ModuleList(
            [FusionExpert(input_dim, expert_hidden_dim, output_dim, dropout) for _ in range(num_experts)]
        )
        self.w_gate = torch.nn.Parameter(torch.zeros(input_dim, num_experts), requires_grad=True)
        self.w_noise = torch.nn.Parameter(torch.zeros(input_dim, num_experts), requires_grad=True)
        self.soft_plus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=1)
        self.register_buffer("normal_mean", torch.tensor([0.0]))
        self.register_buffer("normal_std", torch.tensor([1.0]))

    def noisy_top_k_gating(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        logits = x @ self.w_gate
        if training:
            logits = logits + torch.randn_like(logits) * (self.soft_plus(x @ self.w_noise) + 1e-2)
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(self.top_k, dim=1)
        top_k_gates = top_logits / (top_logits.sum(1, keepdim=True) + 1e-6)
        zeros = torch.zeros_like(logits)
        return zeros.scatter(1, top_indices, top_k_gates)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gates = self.noisy_top_k_gating(x, self.training)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        fused = torch.sum(expert_outputs * gates.unsqueeze(1).expand_as(expert_outputs), dim=2)
        return fused, torch.tensor(0.0, device=x.device), gates


class KANHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, grid_size: int = 8, dropout: float = 0.2, out_dim: int = 2):
        super().__init__()
        self.kan1 = FourierKANLayer(input_dim, hidden_dim, grid_size=grid_size)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        self.kan2 = FourierKANLayer(hidden_dim, out_dim, grid_size=grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.kan1(x)))
        return self.kan2(x)


class DualBranchMoEKAN(torch.nn.Module):
    def __init__(
        self,
        g_dim: int,
        pe_dim: int,
        g_projection_dim: int = 128,
        pe_projection_dim: int = 64,
        fusion_dim: int = 128,
        expert_hidden_dim: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        kan_hidden: int = 64,
        grid_size: int = 8,
        dropout: float = 0.2,
        gate_loss_coef: float = 0.05,
    ):
        super().__init__()
        self.g_encoder = BranchEncoder(g_dim, g_projection_dim, dropout)
        self.pe_encoder = BranchEncoder(pe_dim, pe_projection_dim, dropout)
        fused_input_dim = g_projection_dim + pe_projection_dim
        self.fusion = NoisyTopKGating(
            input_dim=fused_input_dim,
            expert_hidden_dim=expert_hidden_dim,
            output_dim=fusion_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            loss_coef=gate_loss_coef,
        )
        self.head = KANHead(fusion_dim, hidden_dim=kan_hidden, grid_size=grid_size, dropout=dropout, out_dim=2)

    def forward(self, x_g: torch.Tensor, x_pe: torch.Tensor) -> torch.Tensor:
        z_g = self.g_encoder(x_g)
        z_pe = self.pe_encoder(x_pe)
        z = torch.cat([z_g, z_pe], dim=1)
        fused, _, _ = self.fusion(z)
        return self.head(fused)


def balanced_sample_indices(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= len(y):
        return np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes = sorted(np.unique(y).tolist())
    per_class = max_samples // len(classes)
    chosen = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        n = min(len(idx), per_class)
        chosen.append(rng.choice(idx, size=n, replace=False))
    out = np.concatenate(chosen)
    if len(out) < max_samples:
        rest = np.setdiff1d(np.arange(len(y)), out, assume_unique=False)
        add = rng.choice(rest, size=min(len(rest), max_samples - len(out)), replace=False)
        out = np.concatenate([out, add])
    rng.shuffle(out)
    return out.astype(np.int64)


def gradient_importance_single_input(
    model: torch.nn.Module,
    x_scaled: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    max_samples: int,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    idx = balanced_sample_indices(y, max_samples, seed)
    grad_sum = np.zeros(x_scaled.shape[1], dtype=np.float64)
    grad_x_sum = np.zeros(x_scaled.shape[1], dtype=np.float64)
    n_seen = 0
    for start in range(0, len(idx), batch_size):
        rows = idx[start : start + batch_size]
        xb = torch.tensor(x_scaled[rows], dtype=torch.float32, device=device, requires_grad=True)
        logits = model(xb)
        score = logits[:, 1].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        grad = xb.grad.detach().abs().cpu().numpy()
        x_np = xb.detach().cpu().numpy()
        grad_sum += grad.sum(axis=0)
        grad_x_sum += (grad * np.abs(x_np)).sum(axis=0)
        n_seen += len(rows)
    return (grad_sum / max(1, n_seen)).astype(np.float32), (grad_x_sum / max(1, n_seen)).astype(np.float32)


def gradient_importance_dualbranch_g(
    model: torch.nn.Module,
    xg_scaled: np.ndarray,
    xpe_scaled: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    max_samples: int,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    idx = balanced_sample_indices(y, max_samples, seed)
    grad_sum = np.zeros(xg_scaled.shape[1], dtype=np.float64)
    grad_x_sum = np.zeros(xg_scaled.shape[1], dtype=np.float64)
    n_seen = 0
    for start in range(0, len(idx), batch_size):
        rows = idx[start : start + batch_size]
        xg = torch.tensor(xg_scaled[rows], dtype=torch.float32, device=device, requires_grad=True)
        xpe = torch.tensor(xpe_scaled[rows], dtype=torch.float32, device=device)
        logits = model(xg, xpe)
        score = logits[:, 1].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        grad = xg.grad.detach().abs().cpu().numpy()
        x_np = xg.detach().cpu().numpy()
        grad_sum += grad.sum(axis=0)
        grad_x_sum += (grad * np.abs(x_np)).sum(axis=0)
        n_seen += len(rows)
    return (grad_sum / max(1, n_seen)).astype(np.float32), (grad_x_sum / max(1, n_seen)).astype(np.float32)


def rankify(df: pd.DataFrame, score_cols: Sequence[str], prior_gene: str, prior_weight: float) -> pd.DataFrame:
    out = df.copy()
    n = max(1, len(out))
    percentile_cols = []
    for col in score_cols:
        rank_col = f"{col}_rank"
        pct_col = f"{col}_pct"
        out[rank_col] = out[col].rank(method="min", ascending=False, na_option="bottom").astype(int)
        out[pct_col] = 1.0 - (out[rank_col] - 1.0) / max(1.0, n - 1.0)
        percentile_cols.append(pct_col)
    out["model_only_consensus"] = out[percentile_cols].mean(axis=1)
    out["model_only_rank"] = out["model_only_consensus"].rank(method="min", ascending=False).astype(int)
    out["wetlab_prior_SLC1A2"] = out["gene"].eq(prior_gene).astype(float)
    out["prior_aware_consensus"] = (
        (1.0 - prior_weight) * out["model_only_consensus"] + prior_weight * out["wetlab_prior_SLC1A2"]
    )
    out["prior_aware_rank"] = out["prior_aware_consensus"].rank(method="min", ascending=False).astype(int)
    return out


def feature_report_for_gene(
    feature_meta: pd.DataFrame,
    gene: str,
    feature_scores: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    rows = feature_meta[feature_meta["gene"].eq(gene)].copy()
    for name, values in feature_scores.items():
        rows[name] = np.asarray(values)[rows["feature_index"].to_numpy(dtype=np.int64)]
    rows.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="ALS G-only gene-level interpretability")
    parser.add_argument("--g-pkl", type=Path, default=root / "data/human/ALS/genotype_gene_pca.pkl")
    parser.add_argument("--pe-txt", type=Path, default=root / "data/human/ALS/predicted_expression.tsv")
    parser.add_argument("--gonly-checkpoint", type=Path, default=root / "checkpoints/human/ALS/g_only_kang2p.pt")
    parser.add_argument("--dualbranch-checkpoint", type=Path, default=root / "checkpoints/human/ALS/g_pe_dualbranch.pt")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/human_gradient/ALS/g_only")
    parser.add_argument("--target-gene", default="SLC1A2")
    parser.add_argument("--prior-weight", type=float, default=0.50)
    parser.add_argument("--max-grad-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-dualbranch-gradient", action="store_true")
    parser.add_argument("--skip-gonly-gradient", action="store_true")
    parser.add_argument("--run-sgd", action="store_true", help="Also fit a high-dimensional SGD logistic baseline for coefficients.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x_df, y = load_genotype_pkl(args.g_pkl)
    gene_index = GeneIndex.from_columns(x_df.columns)
    x = x_df.values.astype(np.float32, copy=False)
    print(
        f"Loaded G matrix: samples={x.shape[0]}, features={x.shape[1]}, genes={len(gene_index.genes)}, "
        f"class_counts={np.bincount(y).tolist()}",
        flush=True,
    )

    score_frames = []
    feature_scores: Dict[str, np.ndarray] = {}
    method_score_cols: List[str] = []

    smd_signed = standardized_mean_difference(x, y)
    smd_abs = np.abs(smd_signed)
    feature_scores["univariate_smd_abs"] = smd_abs
    score_frames.append(aggregate_feature_scores(smd_abs, gene_index, suffix="univariate_smd_abs", signed_scores=smd_signed))
    method_score_cols.append("univariate_smd_abs_rms")
    print("Finished univariate standardized-mean-difference scores.", flush=True)

    if args.run_sgd:
        print("Fitting high-dimensional SGD logistic baseline...", flush=True)
        x_z, z_mean, z_std = robust_scale_train_matrix(x)
        coef_signed = train_sgd_logistic_coefficients(x_z, y, SEED)
        del x_z, z_mean, z_std
        gc.collect()
        coef_abs = np.abs(coef_signed)
        feature_scores["sgd_logistic_abscoef"] = coef_abs
        score_frames.append(aggregate_feature_scores(coef_abs, gene_index, suffix="sgd_logistic_abscoef", signed_scores=coef_signed))
        method_score_cols.append("sgd_logistic_abscoef_rms")
        print("Finished SGD logistic coefficient scores.", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using torch device: {device}", flush=True)

    if args.gonly_checkpoint.exists():
        ckpt = torch.load(args.gonly_checkpoint, map_location="cpu", weights_only=False)
        best = ckpt.get("best_params", {})
        gonly_model = KANG2PNet(
            input_dim=int(ckpt["input_dim"]),
            projection_dim=int(best.get("kan_projection_dim", 192)),
            kan_hidden=int(best.get("kan_hidden", 96)),
            grid_size=int(best.get("grid_size", 8)),
            dropout=float(best.get("dropout", 0.15)),
        )
        gonly_model.load_state_dict(ckpt["kan_state_dict"])
        gonly_model.to(device)
        gonly_scaled = ((x - ckpt["scaler_mean"]) / ckpt["scaler_scale"]).astype(np.float32, copy=False)
        project_weight = ckpt["kan_state_dict"]["project.0.weight"].detach().cpu().numpy()
        project_norm = np.sqrt(np.mean(project_weight ** 2, axis=0)).astype(np.float32)
        feature_scores["gonly_kan_project_weight"] = project_norm
        score_frames.append(aggregate_feature_scores(project_norm, gene_index, suffix="gonly_kan_project_weight"))
        method_score_cols.append("gonly_kan_project_weight_rms")
        print("Finished G-only KAN projection-weight scores.", flush=True)
        if not args.skip_gonly_gradient:
            print("Computing G-only KAN gradients...", flush=True)
            grad, grad_x = gradient_importance_single_input(
                gonly_model, gonly_scaled, y, device, args.max_grad_samples, args.batch_size, SEED
            )
            feature_scores["gonly_kan_gradient"] = grad
            feature_scores["gonly_kan_grad_x_input"] = grad_x
            score_frames.append(aggregate_feature_scores(grad, gene_index, suffix="gonly_kan_gradient"))
            score_frames.append(aggregate_feature_scores(grad_x, gene_index, suffix="gonly_kan_grad_x_input"))
            method_score_cols.extend(["gonly_kan_gradient_rms", "gonly_kan_grad_x_input_rms"])
            print("Finished G-only KAN gradient scores.", flush=True)

    if args.dualbranch_checkpoint.exists():
        dckpt = torch.load(args.dualbranch_checkpoint, map_location="cpu", weights_only=False)
        # The saved ALS full checkpoint uses the full-size default config.
        dual_model = DualBranchMoEKAN(
            g_dim=int(dckpt["g_dim"]),
            pe_dim=int(dckpt["pe_dim"]),
            g_projection_dim=128,
            pe_projection_dim=64,
            fusion_dim=128,
            expert_hidden_dim=128,
            num_experts=4,
            top_k=2,
            kan_hidden=64,
            grid_size=8,
            dropout=0.2,
            gate_loss_coef=0.05,
        )
        try:
            dual_model.load_state_dict(dckpt["model_state_dict"])
        except RuntimeError:
            dual_model = DualBranchMoEKAN(
                g_dim=int(dckpt["g_dim"]),
                pe_dim=int(dckpt["pe_dim"]),
                g_projection_dim=64,
                pe_projection_dim=32,
                fusion_dim=64,
                expert_hidden_dim=64,
                num_experts=4,
                top_k=2,
                kan_hidden=32,
                grid_size=4,
                dropout=0.2,
                gate_loss_coef=0.05,
            )
            dual_model.load_state_dict(dckpt["model_state_dict"])
        dual_model.to(device)
        g_encoder_weight = dckpt["model_state_dict"]["g_encoder.net.0.weight"].detach().cpu().numpy()
        g_encoder_norm = np.sqrt(np.mean(g_encoder_weight ** 2, axis=0)).astype(np.float32)
        feature_scores["dualbranch_g_encoder_weight"] = g_encoder_norm
        score_frames.append(aggregate_feature_scores(g_encoder_norm, gene_index, suffix="dualbranch_g_encoder_weight"))
        method_score_cols.append("dualbranch_g_encoder_weight_rms")
        print("Finished DualBranch G-encoder weight scores.", flush=True)
        if not args.skip_dualbranch_gradient and args.pe_txt.exists():
            print("Computing DualBranch G-branch gradients with PE held at observed scaled values...", flush=True)
            pe = load_pe_txt(args.pe_txt)
            xg_scaled = ((x - dckpt["g_scaler_mean"]) / dckpt["g_scaler_scale"]).astype(np.float32, copy=False)
            xpe_scaled = ((pe - dckpt["pe_scaler_mean"]) / dckpt["pe_scaler_scale"]).astype(np.float32, copy=False)
            grad, grad_x = gradient_importance_dualbranch_g(
                dual_model, xg_scaled, xpe_scaled, y, device, args.max_grad_samples, args.batch_size, SEED + 7
            )
            feature_scores["dualbranch_g_gradient"] = grad
            feature_scores["dualbranch_g_grad_x_input"] = grad_x
            score_frames.append(aggregate_feature_scores(grad, gene_index, suffix="dualbranch_g_gradient"))
            score_frames.append(aggregate_feature_scores(grad_x, gene_index, suffix="dualbranch_g_grad_x_input"))
            method_score_cols.extend(["dualbranch_g_gradient_rms", "dualbranch_g_grad_x_input_rms"])
            print("Finished DualBranch G-branch gradient scores.", flush=True)

    gene_scores = merge_gene_score_frames(score_frames)
    gene_scores = rankify(gene_scores, method_score_cols, args.target_gene, args.prior_weight)
    gene_scores = gene_scores.sort_values(["prior_aware_rank", "model_only_rank", "gene"]).reset_index(drop=True)

    all_path = args.out_dir / "als_g_gene_interpretability_all_methods.csv"
    gene_scores.to_csv(all_path, index=False)
    gene_scores.sort_values("model_only_rank").head(100).to_csv(args.out_dir / "top100_model_only_genes.csv", index=False)
    gene_scores.sort_values("prior_aware_rank").head(100).to_csv(args.out_dir / "top100_prior_aware_genes.csv", index=False)
    gene_scores[gene_scores["gene"].eq(args.target_gene)].to_csv(args.out_dir / f"{args.target_gene}_gene_report.csv", index=False)
    feature_report_for_gene(
        gene_index.feature_meta,
        args.target_gene,
        feature_scores,
        args.out_dir / f"{args.target_gene}_feature_level_scores.csv",
    )

    summary = {
        "seed": SEED,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_genes": int(len(gene_index.genes)),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "target_gene": args.target_gene,
        "prior_weight": float(args.prior_weight),
        "device_used": str(device),
        "max_grad_samples": int(args.max_grad_samples),
        "method_score_columns": method_score_cols,
        "all_methods_csv": str(all_path),
    }
    (args.out_dir / "analysis_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    target_row = gene_scores[gene_scores["gene"].eq(args.target_gene)]
    if not target_row.empty:
        row = target_row.iloc[0]
        print(
            f"{args.target_gene}: model_only_rank={int(row['model_only_rank'])}, "
            f"prior_aware_rank={int(row['prior_aware_rank'])}, "
            f"model_only_consensus={row['model_only_consensus']:.4f}, "
            f"prior_aware_consensus={row['prior_aware_consensus']:.4f}"
        )
    print(f"Wrote {all_path}")


if __name__ == "__main__":
    main()
