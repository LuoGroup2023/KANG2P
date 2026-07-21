#!/usr/bin/env python
"""DualBranch MoE-KAN interpretability for ALS G+PE inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    aggregate_feature_scores,
    balanced_sample_indices,
    load_genotype_pkl,
    load_pe_txt,
    seed_all,
    standardized_mean_difference,
)
from als_g_add_ig_lime_rankings import load_dualbranch_model  # noqa: E402


def make_pe_meta(columns: List[str]) -> pd.DataFrame:
    rows = []
    for i, col in enumerate(columns):
        # Keep the full Ensembl ID with version as the ranking key. Stripping
        # versions can create duplicate keys and explode merges.
        gene = str(col)
        rows.append(
            {
                "feature_index": i,
                "feature": str(col),
                "chrom": "",
                "gene": gene,
                "ensembl_base": str(col).split(".")[0],
                "component": str(col),
            }
        )
    return pd.DataFrame(rows)


def make_pe_gene_index(columns: List[str]) -> GeneIndex:
    meta = make_pe_meta(columns)
    genes = meta["gene"].tolist()
    gene_to_indices = {gene: np.array([i], dtype=np.int64) for i, gene in enumerate(genes)}
    gene_chrom = {gene: "" for gene in genes}
    return GeneIndex(meta, genes, gene_to_indices, gene_chrom)


def add_rank_columns(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    n = len(df)
    rank_col = f"{score_col}_rank"
    pct_col = f"{score_col}_pct"
    df[rank_col] = df[score_col].rank(method="min", ascending=False, na_option="bottom").astype(int)
    df[pct_col] = 1.0 - (df[rank_col] - 1.0) / max(1.0, n - 1.0)
    return df


def branch_from_preact(net: torch.nn.Sequential, preact: torch.Tensor) -> torch.Tensor:
    z = net[1](preact)
    z = net[2](z)
    z = net[3](z)
    return z


def dual_logits_from_preact(model: DualBranchMoEKAN, g_preact: torch.Tensor, pe_preact: torch.Tensor) -> torch.Tensor:
    z_g = branch_from_preact(model.g_encoder.net, g_preact)
    z_pe = branch_from_preact(model.pe_encoder.net, pe_preact)
    fused, _, _ = model.fusion(torch.cat([z_g, z_pe], dim=1))
    return model.head(fused)


def gradient_both_branches(
    model: DualBranchMoEKAN,
    xg_scaled: np.ndarray,
    xpe_scaled: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    max_samples: int,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    idx = balanced_sample_indices(y, max_samples, seed)
    g_grad_sum = np.zeros(xg_scaled.shape[1], dtype=np.float64)
    pe_grad_sum = np.zeros(xpe_scaled.shape[1], dtype=np.float64)
    g_gxi_sum = np.zeros(xg_scaled.shape[1], dtype=np.float64)
    pe_gxi_sum = np.zeros(xpe_scaled.shape[1], dtype=np.float64)
    n_seen = 0
    for start in range(0, len(idx), batch_size):
        rows = idx[start : start + batch_size]
        xg = torch.tensor(xg_scaled[rows], dtype=torch.float32, device=device, requires_grad=True)
        xpe = torch.tensor(xpe_scaled[rows], dtype=torch.float32, device=device, requires_grad=True)
        logits = model(xg, xpe)
        score = logits[:, 1].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        g_grad = xg.grad.detach().abs()
        pe_grad = xpe.grad.detach().abs()
        g_grad_sum += g_grad.sum(dim=0).cpu().numpy()
        pe_grad_sum += pe_grad.sum(dim=0).cpu().numpy()
        g_gxi_sum += (g_grad * xg.detach().abs()).sum(dim=0).cpu().numpy()
        pe_gxi_sum += (pe_grad * xpe.detach().abs()).sum(dim=0).cpu().numpy()
        n_seen += len(rows)
    denom = max(1, n_seen)
    return (
        (g_grad_sum / denom).astype(np.float32),
        (g_gxi_sum / denom).astype(np.float32),
        (pe_grad_sum / denom).astype(np.float32),
        (pe_gxi_sum / denom).astype(np.float32),
    )


def joint_integrated_gradients(
    model: DualBranchMoEKAN,
    xg_scaled: np.ndarray,
    xpe_scaled: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    max_samples: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    idx = balanced_sample_indices(y, max_samples, seed)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]
    g_ig_sum = np.zeros(xg_scaled.shape[1], dtype=np.float64)
    pe_ig_sum = np.zeros(xpe_scaled.shape[1], dtype=np.float64)
    n_seen = 0
    for start in range(0, len(idx), batch_size):
        rows = idx[start : start + batch_size]
        xg = torch.tensor(xg_scaled[rows], dtype=torch.float32, device=device)
        xpe = torch.tensor(xpe_scaled[rows], dtype=torch.float32, device=device)
        sg = (alphas[:, None, None] * xg[None, :, :]).reshape(-1, xg.shape[1])
        spe = (alphas[:, None, None] * xpe[None, :, :]).reshape(-1, xpe.shape[1])
        sg.requires_grad_(True)
        spe.requires_grad_(True)
        logits = model(sg, spe)
        score = logits[:, 1].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        g_grads = sg.grad.detach().reshape(steps, len(rows), -1).mean(dim=0)
        pe_grads = spe.grad.detach().reshape(steps, len(rows), -1).mean(dim=0)
        g_ig_sum += (xg * g_grads).abs().sum(dim=0).cpu().numpy()
        pe_ig_sum += (xpe * pe_grads).abs().sum(dim=0).cpu().numpy()
        n_seen += len(rows)
    denom = max(1, n_seen)
    return (g_ig_sum / denom).astype(np.float32), (pe_ig_sum / denom).astype(np.float32)


def contribution_matrix(
    x_sample: np.ndarray,
    linear_weight: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    contrib_feature = linear_weight.T * x_sample[:, None]
    contrib_group = np.zeros((n_groups, linear_weight.shape[0]), dtype=np.float32)
    np.add.at(contrib_group, group_codes, contrib_feature.astype(np.float32, copy=False))
    return contrib_group


@torch.no_grad()
def dualbranch_lime_group_ridge(
    model: DualBranchMoEKAN,
    xg_scaled: np.ndarray,
    xpe_scaled: np.ndarray,
    y: np.ndarray,
    g_index: GeneIndex,
    pe_index: GeneIndex,
    device: torch.device,
    n_local_samples: int,
    n_perturbations: int,
    keep_prob: float,
    ridge_lambda: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = balanced_sample_indices(y, n_local_samples, seed + 9)
    g_genes = g_index.genes
    pe_genes = pe_index.genes
    g_codes = g_index.feature_meta["gene"].map({g: i for i, g in enumerate(g_genes)}).to_numpy(dtype=np.int64)
    pe_codes = pe_index.feature_meta["gene"].map({g: i for i, g in enumerate(pe_genes)}).to_numpy(dtype=np.int64)
    n_g = len(g_genes)
    n_pe = len(pe_genes)
    n_total = n_g + n_pe

    g_linear = model.g_encoder.net[0]
    pe_linear = model.pe_encoder.net[0]
    gw = g_linear.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    pew = pe_linear.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    gb = g_linear.bias.detach().to(device)
    peb = pe_linear.bias.detach().to(device)
    coef_sum = np.zeros(n_total, dtype=np.float64)

    for sample_i, row in enumerate(idx, start=1):
        g_contrib = contribution_matrix(xg_scaled[row], gw, g_codes, n_g)
        pe_contrib = contribution_matrix(xpe_scaled[row], pew, pe_codes, n_pe)
        g_contrib_t = torch.tensor(g_contrib, dtype=torch.float32, device=device)
        pe_contrib_t = torch.tensor(pe_contrib, dtype=torch.float32, device=device)
        g_orig = g_contrib_t.sum(dim=0, keepdim=True) + gb.view(1, -1)
        pe_orig = pe_contrib_t.sum(dim=0, keepdim=True) + peb.view(1, -1)

        masks_np = rng.binomial(1, keep_prob, size=(n_perturbations, n_total)).astype(np.float32)
        masks_np[0, :] = 1.0
        masks_np[1, :] = 0.0
        masks = torch.tensor(masks_np, dtype=torch.float32, device=device)
        removed = 1.0 - masks
        rg = removed[:, :n_g]
        rpe = removed[:, n_g:]
        g_preact = g_orig - rg @ g_contrib_t
        pe_preact = pe_orig - rpe @ pe_contrib_t
        logits = dual_logits_from_preact(model, g_preact, pe_preact)
        response = torch.softmax(logits, dim=1)[:, 1]
        x_design = masks - masks.mean(dim=0, keepdim=True)
        y_centered = response - response.mean()
        kernel = x_design @ x_design.T
        eye = torch.eye(n_perturbations, dtype=torch.float32, device=device)
        alpha = torch.linalg.solve(kernel + ridge_lambda * eye, y_centered[:, None]).squeeze(1)
        coef = x_design.T @ alpha
        coef_sum += coef.detach().abs().cpu().numpy()
        print(f"  G+PE LIME local surrogate {sample_i}/{len(idx)} done", flush=True)

    coef = (coef_sum / max(1, len(idx))).astype(np.float32)
    return coef[:n_g], coef[n_g:]


def score_frame(
    index: GeneIndex,
    scores: Dict[str, np.ndarray],
    modality: str,
) -> pd.DataFrame:
    frames = []
    for name, values in scores.items():
        values = np.asarray(values)
        if values.shape[0] == len(index.genes):
            frames.append(
                pd.DataFrame(
                    {
                        "gene": index.genes,
                        "chrom": [index.gene_chrom.get(gene, "") for gene in index.genes],
                        "n_features": [len(index.gene_to_indices[gene]) for gene in index.genes],
                        f"{name}_max": values,
                        f"{name}_mean": values,
                        f"{name}_rms": values,
                        f"{name}_sum": values,
                        f"{name}_signed_at_max": values,
                        f"{name}_top_feature": index.genes,
                    }
                )
            )
        else:
            frames.append(aggregate_feature_scores(values, index, suffix=name))
    out = frames[0][["gene", "chrom", "n_features"]].copy()
    for frame in frames:
        keep = ["gene"] + [c for c in frame.columns if c.startswith(tuple(scores.keys()))]
        out = out.merge(frame[keep], on="gene", how="left")
    out.insert(0, "modality", modality)
    for name in scores:
        out = add_rank_columns(out, f"{name}_rms")
    return out


def export_rankings(df: pd.DataFrame, out_dir: Path, modality: str) -> None:
    rank_dir = out_dir / f"{modality}_per_method_rankings"
    rank_dir.mkdir(parents=True, exist_ok=True)
    for col in sorted([c for c in df.columns if c.endswith("_rank")]):
        base = col[: -len("_rank")]
        score_col = base
        cols = ["modality", "gene", "chrom", "n_features", col]
        if score_col in df.columns:
            cols.append(score_col)
        pct = f"{base}_pct"
        if pct in df.columns:
            cols.append(pct)
        df[cols].sort_values(col).rename(columns={col: "rank"}).to_csv(rank_dir / f"{base}.csv", index=False)


def slc_summary(g_df: pd.DataFrame, pe_df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for col in sorted([c for c in g_df.columns if c.endswith("_rank")]):
        base = col[: -len("_rank")]
        hit = g_df[g_df["gene"].eq(target)]
        rows.append(
            {
                "method": base,
                "modality": "G",
                "target_gene": target,
                "target_present": not hit.empty,
                "target_rank": int(hit.iloc[0][col]) if not hit.empty else np.nan,
                "top_gene_or_id": str(g_df.sort_values(col).iloc[0]["gene"]),
            }
        )
    target_ensg = "ENSG00000110436"
    for col in sorted([c for c in pe_df.columns if c.endswith("_rank")]):
        base = col[: -len("_rank")]
        hit = pe_df[pe_df["gene"].astype(str).str.startswith(target_ensg)]
        rows.append(
            {
                "method": base,
                "modality": "PE",
                "target_gene": f"{target}/{target_ensg}",
                "target_present": not hit.empty,
                "target_rank": int(hit.iloc[0][col]) if not hit.empty else np.nan,
                "top_gene_or_id": str(pe_df.sort_values(col).iloc[0]["gene"]),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="ALS G+PE DualBranch gene interpretability")
    parser.add_argument("--g-pkl", type=Path, default=root / "data/human/ALS/genotype_gene_pca.pkl")
    parser.add_argument("--pe-txt", type=Path, default=root / "data/human/ALS/predicted_expression.tsv")
    parser.add_argument("--dualbranch-checkpoint", type=Path, default=root / "checkpoints/human/ALS/g_pe_dualbranch.pt")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/human_gradient/ALS/g_pe")
    parser.add_argument("--target-gene", default="SLC1A2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grad-samples", type=int, default=4096)
    parser.add_argument("--grad-batch-size", type=int, default=128)
    parser.add_argument("--ig-samples", type=int, default=2048)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--ig-batch-size", type=int, default=24)
    parser.add_argument("--lime-local-samples", type=int, default=4)
    parser.add_argument("--lime-perturbations", type=int, default=768)
    parser.add_argument("--lime-keep-prob", type=float, default=0.85)
    parser.add_argument("--lime-ridge-lambda", type=float, default=10.0)
    parser.add_argument("--skip-lime", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("Loading G and PE inputs...", flush=True)
    g_df, y = load_genotype_pkl(args.g_pkl)
    pe_raw = pd.read_csv(args.pe_txt, sep="\t")
    pe_cols = list(pe_raw.columns)
    pe = pe_raw.values.astype(np.float32, copy=False)
    g = g_df.values.astype(np.float32, copy=False)
    g_index = GeneIndex.from_columns(g_df.columns)
    pe_index = make_pe_gene_index(pe_cols)
    print(f"G={g.shape}, PE={pe.shape}, G genes={len(g_index.genes)}, PE ids={len(pe_index.genes)}, device={device}", flush=True)

    model, ckpt = load_dualbranch_model(args.dualbranch_checkpoint, device)
    xg = ((g - ckpt["g_scaler_mean"]) / ckpt["g_scaler_scale"]).astype(np.float32, copy=False)
    xpe = ((pe - ckpt["pe_scaler_mean"]) / ckpt["pe_scaler_scale"]).astype(np.float32, copy=False)

    g_scores: Dict[str, np.ndarray] = {}
    pe_scores: Dict[str, np.ndarray] = {}
    print("Computing univariate branch scores...", flush=True)
    g_scores["univariate_smd_abs"] = np.abs(standardized_mean_difference(g, y))
    pe_scores["univariate_smd_abs"] = np.abs(standardized_mean_difference(pe, y))

    print("Computing branch encoder weight scores...", flush=True)
    g_w = ckpt["model_state_dict"]["g_encoder.net.0.weight"].detach().cpu().numpy()
    pe_w = ckpt["model_state_dict"]["pe_encoder.net.0.weight"].detach().cpu().numpy()
    g_scores["dualbranch_encoder_weight"] = np.sqrt(np.mean(g_w ** 2, axis=0)).astype(np.float32)
    pe_scores["dualbranch_encoder_weight"] = np.sqrt(np.mean(pe_w ** 2, axis=0)).astype(np.float32)

    print("Computing branch gradients...", flush=True)
    g_grad, g_gxi, pe_grad, pe_gxi = gradient_both_branches(
        model, xg, xpe, y, device, args.grad_samples, args.grad_batch_size, SEED + 404
    )
    g_scores["dualbranch_gradient"] = g_grad
    g_scores["dualbranch_grad_x_input"] = g_gxi
    pe_scores["dualbranch_gradient"] = pe_grad
    pe_scores["dualbranch_grad_x_input"] = pe_gxi

    print("Computing joint Integrated Gradients...", flush=True)
    g_ig, pe_ig = joint_integrated_gradients(
        model, xg, xpe, y, device, args.ig_samples, args.ig_batch_size, args.ig_steps, SEED + 505
    )
    g_scores["dualbranch_joint_integrated_gradients"] = g_ig
    pe_scores["dualbranch_joint_integrated_gradients"] = pe_ig

    if not args.skip_lime:
        print("Computing joint G+PE gene-group LIME-style surrogate...", flush=True)
        g_lime, pe_lime = dualbranch_lime_group_ridge(
            model,
            xg,
            xpe,
            y,
            g_index,
            pe_index,
            device,
            args.lime_local_samples,
            args.lime_perturbations,
            args.lime_keep_prob,
            args.lime_ridge_lambda,
            SEED + 606,
        )
        g_scores["dualbranch_gpe_lime_gene_ridge_abscoef"] = g_lime
        pe_scores["dualbranch_gpe_lime_gene_ridge_abscoef"] = pe_lime

    g_rank = score_frame(g_index, g_scores, "G")
    pe_rank = score_frame(pe_index, pe_scores, "PE")
    g_rank.to_csv(args.out_dir / "g_branch_gene_rankings.csv", index=False)
    pe_rank.to_csv(args.out_dir / "pe_branch_ensembl_rankings.csv", index=False)
    export_rankings(g_rank, args.out_dir, "G")
    export_rankings(pe_rank, args.out_dir, "PE")
    summary = slc_summary(g_rank, pe_rank, args.target_gene)
    summary.to_csv(args.out_dir / f"{args.target_gene}_gpe_per_method_rank_summary.csv", index=False)

    config = {
        "device_used": str(device),
        "g_shape": list(g.shape),
        "pe_shape": list(pe.shape),
        "pe_note": "PE features are Ensembl IDs from All_brain_MRMR_exp.txt. No local Ensembl-to-symbol mapping was found.",
        "target_gene": args.target_gene,
        "target_pe_ensembl_checked": "ENSG00000110436",
        "target_present_in_pe": bool((pe_rank["gene"] == "ENSG00000110436").any()),
        "grad_samples": args.grad_samples,
        "ig_samples": args.ig_samples,
        "ig_steps": args.ig_steps,
        "lime_local_samples": args.lime_local_samples,
        "lime_perturbations": args.lime_perturbations,
    }
    (args.out_dir / "gpe_interpretability_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote G+PE DualBranch rankings to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
