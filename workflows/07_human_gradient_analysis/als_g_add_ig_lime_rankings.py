#!/usr/bin/env python
"""Add Integrated Gradients, gene-group LIME, and per-method rankings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from als_g_gene_interpretability import (  # noqa: E402
    SEED,
    DualBranchMoEKAN,
    GeneIndex,
    KANG2PNet,
    aggregate_feature_scores,
    balanced_sample_indices,
    load_genotype_pkl,
    load_pe_txt,
    seed_all,
)


def gonly_logits_from_project_preact(model: KANG2PNet, preact: torch.Tensor) -> torch.Tensor:
    z = model.project[1](preact)
    z = model.project[2](z)
    z = model.project[3](z)
    z = model.drop(model.act(model.kan1(z)))
    return model.kan2(z)


def load_gonly_model(checkpoint_path: Path, input_dim: int, device: torch.device) -> Tuple[KANG2PNet, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best = ckpt.get("best_params", {})
    model = KANG2PNet(
        input_dim=input_dim,
        projection_dim=int(best.get("kan_projection_dim", 192)),
        kan_hidden=int(best.get("kan_hidden", 96)),
        grid_size=int(best.get("grid_size", 8)),
        dropout=float(best.get("dropout", 0.15)),
    )
    model.load_state_dict(ckpt["kan_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def load_dualbranch_model(checkpoint_path: Path, device: torch.device) -> Tuple[DualBranchMoEKAN, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tried = [
        dict(g_projection_dim=128, pe_projection_dim=64, fusion_dim=128, expert_hidden_dim=128, kan_hidden=64, grid_size=8),
        dict(g_projection_dim=64, pe_projection_dim=32, fusion_dim=64, expert_hidden_dim=64, kan_hidden=32, grid_size=4),
    ]
    last_err = None
    for cfg in tried:
        model = DualBranchMoEKAN(
            g_dim=int(ckpt["g_dim"]),
            pe_dim=int(ckpt["pe_dim"]),
            num_experts=4,
            top_k=2,
            dropout=0.2,
            gate_loss_coef=0.05,
            **cfg,
        )
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            return model, ckpt
        except RuntimeError as err:
            last_err = err
    raise RuntimeError(f"Could not load DualBranch checkpoint: {last_err}")


def integrated_gradients_single_input(
    model: KANG2PNet,
    x_scaled: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    max_samples: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    idx = balanced_sample_indices(y, max_samples, seed)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]
    ig_sum = np.zeros(x_scaled.shape[1], dtype=np.float64)
    n_seen = 0
    for start in range(0, len(idx), batch_size):
        rows = idx[start : start + batch_size]
        xb = torch.tensor(x_scaled[rows], dtype=torch.float32, device=device)
        scaled = (alphas[:, None, None] * xb[None, :, :]).reshape(-1, xb.shape[1])
        scaled.requires_grad_(True)
        logits = model(scaled)
        score = logits[:, 1].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        grads = scaled.grad.detach().reshape(steps, len(rows), -1).mean(dim=0)
        ig = xb * grads
        ig_sum += ig.detach().abs().sum(dim=0).cpu().numpy()
        n_seen += len(rows)
    return (ig_sum / max(1, n_seen)).astype(np.float32)


def integrated_gradients_dualbranch_g(
    model: DualBranchMoEKAN,
    xg_scaled: np.ndarray,
    xpe_scaled: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    max_samples: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    idx = balanced_sample_indices(y, max_samples, seed)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]
    ig_sum = np.zeros(xg_scaled.shape[1], dtype=np.float64)
    n_seen = 0
    for start in range(0, len(idx), batch_size):
        rows = idx[start : start + batch_size]
        xg = torch.tensor(xg_scaled[rows], dtype=torch.float32, device=device)
        xpe = torch.tensor(xpe_scaled[rows], dtype=torch.float32, device=device)
        scaled_g = (alphas[:, None, None] * xg[None, :, :]).reshape(-1, xg.shape[1])
        scaled_pe = xpe.repeat(steps, 1)
        scaled_g.requires_grad_(True)
        logits = model(scaled_g, scaled_pe)
        score = logits[:, 1].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        grads = scaled_g.grad.detach().reshape(steps, len(rows), -1).mean(dim=0)
        ig = xg * grads
        ig_sum += ig.detach().abs().sum(dim=0).cpu().numpy()
        n_seen += len(rows)
    return (ig_sum / max(1, n_seen)).astype(np.float32)


def add_rank_columns(df: pd.DataFrame, score_col: str, *, suffix: str | None = None) -> pd.DataFrame:
    suffix = suffix or score_col
    n = len(df)
    rank_col = f"{suffix}_rank"
    pct_col = f"{suffix}_pct"
    df[rank_col] = df[score_col].rank(method="min", ascending=False, na_option="bottom").astype(int)
    df[pct_col] = 1.0 - (df[rank_col] - 1.0) / max(1.0, n - 1.0)
    return df


def gene_component_contribution_matrix(
    x_sample: np.ndarray,
    linear_weight: np.ndarray,
    gene_codes: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    # contribution[gene, projection_dim] = sum_j x_j * W[:, j] for features j in gene
    contrib_feature = linear_weight.T * x_sample[:, None]
    contrib_gene = np.zeros((n_genes, linear_weight.shape[0]), dtype=np.float32)
    np.add.at(contrib_gene, gene_codes, contrib_feature.astype(np.float32, copy=False))
    return contrib_gene


@torch.no_grad()
def lime_gene_group_ridge(
    model: KANG2PNet,
    x_scaled: np.ndarray,
    y: np.ndarray,
    gene_index: GeneIndex,
    device: torch.device,
    n_local_samples: int,
    n_perturbations: int,
    keep_prob: float,
    ridge_lambda: float,
    seed: int,
) -> np.ndarray:
    """LIME-style ridge surrogate over gene-group binary masks.

    This perturbs complete gene component groups before the first KAN projection
    layer and fits a local ridge surrogate in mask space. It returns global gene
    importance as the mean absolute local coefficient over sampled individuals.
    """
    rng = np.random.default_rng(seed)
    idx = balanced_sample_indices(y, n_local_samples, seed + 17)
    genes = gene_index.genes
    gene_to_code = {gene: i for i, gene in enumerate(genes)}
    gene_codes = gene_index.feature_meta["gene"].map(gene_to_code).to_numpy(dtype=np.int64)
    n_genes = len(genes)

    linear = model.project[0]
    weight_np = linear.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    bias = linear.bias.detach().to(device)

    coef_sum = np.zeros(n_genes, dtype=np.float64)
    for sample_i, row in enumerate(idx, start=1):
        x_sample = x_scaled[row]
        contrib_gene = gene_component_contribution_matrix(x_sample, weight_np, gene_codes, n_genes)
        contrib_t = torch.tensor(contrib_gene, dtype=torch.float32, device=device)
        original_preact = contrib_t.sum(dim=0, keepdim=True) + bias.view(1, -1)

        masks_np = rng.binomial(1, keep_prob, size=(n_perturbations, n_genes)).astype(np.float32)
        masks_np[0, :] = 1.0
        masks_np[1, :] = 0.0
        masks = torch.tensor(masks_np, dtype=torch.float32, device=device)
        removed = 1.0 - masks
        preact = original_preact - removed @ contrib_t
        logits = gonly_logits_from_project_preact(model, preact)
        response = torch.softmax(logits, dim=1)[:, 1]

        x_design = masks - masks.mean(dim=0, keepdim=True)
        y_centered = response - response.mean()
        kernel = x_design @ x_design.T
        eye = torch.eye(n_perturbations, dtype=torch.float32, device=device)
        alpha = torch.linalg.solve(kernel + ridge_lambda * eye, y_centered[:, None]).squeeze(1)
        coef = x_design.T @ alpha
        coef_sum += coef.detach().abs().cpu().numpy()
        print(f"  LIME local surrogate {sample_i}/{len(idx)} done", flush=True)

    return (coef_sum / max(1, len(idx))).astype(np.float32)


def export_per_method_rankings(df: pd.DataFrame, out_dir: Path, target_gene: str) -> pd.DataFrame:
    rank_dir = out_dir / "per_method_rankings"
    rank_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rank_cols = [c for c in df.columns if c.endswith("_rank") and c not in {"model_only_rank", "prior_aware_rank"}]
    for rank_col in sorted(rank_cols):
        base = rank_col[: -len("_rank")]
        score_col = base
        if score_col not in df.columns:
            score_col = base.replace("_rms", "_rms")
        pct_col = f"{base}_pct"
        cols = ["gene", "chrom", "n_features", rank_col]
        if score_col in df.columns:
            cols.append(score_col)
        if pct_col in df.columns:
            cols.append(pct_col)
        tab = df[cols].sort_values(rank_col).rename(columns={rank_col: "rank"})
        safe = base.replace("/", "_").replace(" ", "_")
        tab.to_csv(rank_dir / f"{safe}.csv", index=False)
        target = tab[tab["gene"].eq(target_gene)]
        if not target.empty:
            rows.append(
                {
                    "method": base,
                    "target_gene": target_gene,
                    "target_rank": int(target.iloc[0]["rank"]),
                    "top_gene": str(tab.iloc[0]["gene"]),
                    "top_rank_score": float(tab.iloc[0][score_col]) if score_col in tab.columns else np.nan,
                }
            )
    summary = pd.DataFrame(rows).sort_values("target_rank")
    summary.to_csv(out_dir / f"{target_gene}_per_method_rank_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Add IG/LIME and export per-method gene rankings")
    parser.add_argument("--g-pkl", type=Path, default=root / "data/human/ALS/genotype_gene_pca.pkl")
    parser.add_argument("--pe-txt", type=Path, default=root / "data/human/ALS/predicted_expression.tsv")
    parser.add_argument("--base-result-dir", type=Path, default=root / "outputs/human_gradient/ALS/g_only")
    parser.add_argument("--gonly-checkpoint", type=Path, default=root / "checkpoints/human/ALS/g_only_kang2p.pt")
    parser.add_argument("--dualbranch-checkpoint", type=Path, default=root / "checkpoints/human/ALS/g_pe_dualbranch.pt")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/human_gradient/ALS/g_only_ig_lime")
    parser.add_argument("--target-gene", default="SLC1A2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ig-samples", type=int, default=2048)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--ig-batch-size", type=int, default=32)
    parser.add_argument("--lime-local-samples", type=int, default=6)
    parser.add_argument("--lime-perturbations", type=int, default=1024)
    parser.add_argument("--lime-keep-prob", type=float, default=0.85)
    parser.add_argument("--lime-ridge-lambda", type=float, default=10.0)
    parser.add_argument("--skip-lime", action="store_true")
    parser.add_argument("--skip-dualbranch-ig", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    print("Loading genotype data...", flush=True)
    x_df, y = load_genotype_pkl(args.g_pkl)
    gene_index = GeneIndex.from_columns(x_df.columns)
    x = x_df.values.astype(np.float32, copy=False)
    print(f"Loaded G matrix {x.shape}; genes={len(gene_index.genes)}; device={device}", flush=True)

    base_csv = args.base_result_dir / "als_g_gene_interpretability_all_methods.csv"
    df = pd.read_csv(base_csv)
    if "gene" not in df.columns:
        raise ValueError(f"Missing gene column in {base_csv}")

    print("Loading G-only KAN model...", flush=True)
    g_model, g_ckpt = load_gonly_model(args.gonly_checkpoint, x.shape[1], device)
    xg_scaled = ((x - g_ckpt["scaler_mean"]) / g_ckpt["scaler_scale"]).astype(np.float32, copy=False)

    print("Computing G-only Integrated Gradients...", flush=True)
    g_ig = integrated_gradients_single_input(
        g_model, xg_scaled, y, device, args.ig_samples, args.ig_batch_size, args.ig_steps, SEED + 101
    )
    g_ig_gene = aggregate_feature_scores(g_ig, gene_index, suffix="gonly_kan_integrated_gradients")
    df = df.merge(g_ig_gene[["gene", "gonly_kan_integrated_gradients_rms", "gonly_kan_integrated_gradients_max", "gonly_kan_integrated_gradients_top_feature"]], on="gene", how="left")
    df = add_rank_columns(df, "gonly_kan_integrated_gradients_rms")

    if not args.skip_dualbranch_ig:
        print("Loading DualBranch model and PE data...", flush=True)
        d_model, d_ckpt = load_dualbranch_model(args.dualbranch_checkpoint, device)
        pe = load_pe_txt(args.pe_txt)
        xdg_scaled = ((x - d_ckpt["g_scaler_mean"]) / d_ckpt["g_scaler_scale"]).astype(np.float32, copy=False)
        xpe_scaled = ((pe - d_ckpt["pe_scaler_mean"]) / d_ckpt["pe_scaler_scale"]).astype(np.float32, copy=False)
        print("Computing DualBranch G Integrated Gradients...", flush=True)
        d_ig = integrated_gradients_dualbranch_g(
            d_model, xdg_scaled, xpe_scaled, y, device, args.ig_samples, args.ig_batch_size, args.ig_steps, SEED + 202
        )
        d_ig_gene = aggregate_feature_scores(d_ig, gene_index, suffix="dualbranch_g_integrated_gradients")
        df = df.merge(d_ig_gene[["gene", "dualbranch_g_integrated_gradients_rms", "dualbranch_g_integrated_gradients_max", "dualbranch_g_integrated_gradients_top_feature"]], on="gene", how="left")
        df = add_rank_columns(df, "dualbranch_g_integrated_gradients_rms")

    if not args.skip_lime:
        print("Computing gene-group LIME-style ridge surrogate...", flush=True)
        lime = lime_gene_group_ridge(
            g_model,
            xg_scaled,
            y,
            gene_index,
            device,
            args.lime_local_samples,
            args.lime_perturbations,
            args.lime_keep_prob,
            args.lime_ridge_lambda,
            SEED + 303,
        )
        lime_df = pd.DataFrame({"gene": gene_index.genes, "gonly_kan_lime_gene_ridge_abscoef": lime})
        df = df.merge(lime_df, on="gene", how="left")
        df = add_rank_columns(df, "gonly_kan_lime_gene_ridge_abscoef")

    df.to_csv(args.out_dir / "als_g_gene_interpretability_with_ig_lime.csv", index=False)
    summary = export_per_method_rankings(df, args.out_dir, args.target_gene)
    summary.to_csv(args.out_dir / f"{args.target_gene}_per_method_rank_summary.csv", index=False)

    config = {
        "device_used": str(device),
        "ig_samples": args.ig_samples,
        "ig_steps": args.ig_steps,
        "lime_local_samples": args.lime_local_samples,
        "lime_perturbations": args.lime_perturbations,
        "lime_keep_prob": args.lime_keep_prob,
        "lime_ridge_lambda": args.lime_ridge_lambda,
        "lime_note": "Gene-group LIME-style ridge surrogate over binary gene masks before the first KAN projection layer.",
    }
    (args.out_dir / "ig_lime_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(summary.to_string(index=False), flush=True)
    print(f"Wrote augmented rankings to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
