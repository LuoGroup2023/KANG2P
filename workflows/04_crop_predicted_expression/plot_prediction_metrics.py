#!/usr/bin/env python3
"""Plot crop SNP2Expression prediction-accuracy histograms."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _chr_sort_key(path: Path) -> int:
    name = path.parent.name
    if not name.startswith("chr"):
        raise ValueError(f"Unexpected chromosome directory name: {name}")
    return int(name.removeprefix("chr"))


def load_metrics(result_dir: Path, expected_chromosomes: int) -> pd.DataFrame:
    metric_files = sorted(
        result_dir.glob("chr*/Predict_metrics.tsv"),
        key=_chr_sort_key,
    )
    if expected_chromosomes > 0 and len(metric_files) != expected_chromosomes:
        raise RuntimeError(
            f"Expected {expected_chromosomes} Predict_metrics.tsv files, found {len(metric_files)}"
        )
    if not metric_files:
        raise RuntimeError(f"No chr*/Predict_metrics.tsv files found under {result_dir}")

    frames = []
    for metric_file in metric_files:
        chrom = metric_file.parent.name.removeprefix("chr")
        frame = pd.read_csv(metric_file, sep="\t")
        frame.insert(0, "Chr", chrom)
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    required = {"Gene", "R2", "SCC", "MSE"}
    missing = required.difference(metrics.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(sorted(missing))}")
    return metrics


def save_pub(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")


def annotate_quantile(
    ax: plt.Axes,
    q_value: float,
    quantile_label: str,
) -> None:
    y_top = ax.get_ylim()[1]
    x_mid = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    ha = "right" if q_value > x_mid else "left"
    x_pad = 0.015 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    text_x = q_value - x_pad if ha == "right" else q_value + x_pad

    ax.axvline(q_value, color="#9a3412", linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(
        text_x,
        y_top * 0.90,
        f"{quantile_label} quantile\n{q_value:.3f}",
        ha=ha,
        va="top",
        fontsize=6.2,
        color="#7c2d12",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
    )


def plot_panel(
    ax: plt.Axes,
    values: pd.Series,
    metric: str,
    panel_label: str,
    color: str,
    bins: int | np.ndarray,
    xlim: tuple[float, float] | None = None,
    note: str | None = None,
    note_y: float = 0.72,
    quantile: float = 0.90,
    quantile_label: str = "90%",
) -> dict[str, float | int | str]:
    clean = values.dropna().astype(float)
    q_value = float(clean.quantile(quantile))

    plot_values = clean
    if xlim is not None:
        plot_values = clean[(clean >= xlim[0]) & (clean <= xlim[1])]

    counts, _, _ = ax.hist(
        plot_values,
        bins=bins,
        color=color,
        edgecolor="white",
        linewidth=0.35,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_title(metric, loc="left", pad=4, fontsize=7.2, fontweight="bold")
    ax.text(
        -0.20,
        1.10,
        panel_label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.98,
        0.98,
        f"n = {len(clean):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color="#3f3f46",
    )
    annotate_quantile(ax, q_value, quantile_label)

    if note:
        ax.text(
            0.98,
            note_y,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.7,
            color="#52525b",
        )

    ax.set_xlabel(metric)
    ax.set_ylabel("Gene count")
    ax.tick_params(axis="both", labelsize=6.1, length=2.4, width=0.45)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.margins(y=0.12)

    return {
        "metric": metric,
        "n": int(len(clean)),
        "missing": int(values.isna().sum()),
        "min": float(clean.min()),
        "median": float(clean.median()),
        "quantile": quantile,
        "quantile_label": quantile_label,
        "quantile_value": q_value,
        "max": float(clean.max()),
        "histogram_n": int(len(plot_values)),
        "histogram_max_count": int(counts.max()) if len(counts) else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-label", default="crop")
    parser.add_argument(
        "--expected-chromosomes",
        type=int,
        default=0,
        help="Require this many chromosome metric files; 0 accepts any positive count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.result_dir / "figures"
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.dataset_label).strip("_") or "crop"
    output_stem = output_dir / f"{safe_label}_prediction_accuracy_histograms"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.55,
            "axes.labelsize": 6.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 150,
            "savefig.facecolor": "white",
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.result_dir, args.expected_chromosomes)
    metrics = metrics.dropna(subset=["SCC"]).copy()

    q995_mse = float(metrics["MSE"].dropna().quantile(0.995))
    mse_tail_n = int((metrics["MSE"].dropna() > q995_mse).sum())

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.03, wspace=0.08, hspace=0.02)

    summaries = [
        plot_panel(
            axes[0],
            metrics["R2"],
            "R2",
            "a",
            "#4C78A8",
            bins=np.linspace(-0.5, 1.0, 61),
            xlim=(-0.5, 1.0),
            note=f"< -0.5: {int((metrics['R2'] < -0.5).sum()):,}",
            note_y=0.74,
            quantile=0.90,
            quantile_label="90%",
        ),
        plot_panel(
            axes[1],
            metrics["SCC"],
            "SCC",
            "b",
            "#59A14F",
            bins=np.linspace(-0.4, 1.0, 57),
            xlim=(-0.4, 1.0),
            note=None,
            note_y=0.68,
            quantile=0.90,
            quantile_label="90%",
        ),
        plot_panel(
            axes[2],
            metrics["MSE"],
            "MSE",
            "c",
            "#B279A2",
            bins=np.linspace(0, q995_mse, 62),
            xlim=(0, q995_mse),
            note=f"> 99.5%: {mse_tail_n:,}",
            note_y=0.78,
            quantile=0.10,
            quantile_label="10%",
        ),
    ]

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.35)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Distribution of SNP2Expression prediction accuracy across {args.dataset_label} genes",
        x=0.01,
        y=1.08,
        ha="left",
        fontsize=7.4,
        fontweight="bold",
    )

    pd.DataFrame(summaries).to_csv(
        output_dir / f"{safe_label}_prediction_accuracy_quantiles.tsv",
        sep="\t",
        index=False,
    )
    metrics[["Chr", "Gene", "R2", "SCC", "MSE"]].to_csv(
        output_dir / f"{safe_label}_prediction_accuracy_source.tsv",
        sep="\t",
        index=False,
    )
    save_pub(fig, output_stem)
    plt.close(fig)


if __name__ == "__main__":
    main()
