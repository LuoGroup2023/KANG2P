#!/usr/bin/env python3
"""Run the DiseaseCapsule paper-style PLINK QC steps on an ALS PLINK prefix."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_cmd(cmd: list[str], log_path: Path) -> None:
    print("+ " + " ".join(map(str, cmd)), flush=True)
    with log_path.open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def require_plink_files(prefix: Path) -> None:
    for suffix in (".bed", ".bim", ".fam"):
        path = plink_path(prefix, suffix)
        if not path.exists():
            raise FileNotFoundError(path)


def plink_path(prefix: Path, suffix: str) -> Path:
    return Path(str(prefix) + suffix)


def count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def prefix_counts(prefix: Path) -> tuple[int, int]:
    return count_lines(plink_path(prefix, ".bim")), count_lines(plink_path(prefix, ".fam"))


def parse_diff_missing(missing_path: Path, exclude_path: Path, p_threshold: float) -> int:
    missing = pd.read_csv(
        missing_path,
        sep=r"\s+",
        engine="c",
        na_values=["NA", "nan", "NaN"],
    )
    if "SNP" not in missing.columns or "P" not in missing.columns:
        raise ValueError(f"{missing_path} does not look like a PLINK .missing file.")
    pvals = pd.to_numeric(missing["P"], errors="coerce")
    excluded = missing.loc[pvals.notna() & (pvals <= p_threshold), "SNP"].astype(str)
    excluded = excluded.drop_duplicates().sort_values()
    exclude_path.write_text("\n".join(excluded.tolist()) + ("\n" if len(excluded) else ""))
    return int(len(excluded))


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["step", "prefix", "variants", "samples", "notes"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run paper-style ALS PLINK QC.")
    parser.add_argument(
        "--source-prefix",
        type=Path,
        default=root / "data/ALS/02_merged_all_QCedSNPs",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=root / "outputs/ALS/03_merged_all_paperQC_autosome_keepSamples",
    )
    parser.add_argument("--plink-bin", default=os.environ.get("PLINK_BIN", "plink"))
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--diff-missing-p", type=float, default=1e-4)
    parser.add_argument(
        "--mind-threshold",
        default="none",
        help="Sample missingness threshold for --mind. Use 'none' to keep all samples.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    require_plink_files(args.source_prefix)
    if not Path(args.plink_bin).exists() and shutil.which(args.plink_bin) is None:
        raise FileNotFoundError(f"Cannot find PLINK executable: {args.plink_bin}")
    mind_threshold = None if str(args.mind_threshold).lower() in {"", "none", "no", "false", "off", "0"} else str(args.mind_threshold)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    step1_label = "step1_autosome_geno01_keepSamples" if mind_threshold is None else f"step1_autosome_geno01_mind{mind_threshold}"
    step1_label = step1_label.replace(".", "")
    step1 = args.out_prefix.with_name(args.out_prefix.name + f".{step1_label}")
    step2 = args.out_prefix.with_name(args.out_prefix.name + ".step2_strict_snp_qc")
    diff_prefix = args.out_prefix.with_name(args.out_prefix.name + ".step3_diff_missing")
    diff_exclude = args.out_prefix.with_name(args.out_prefix.name + ".step3_diff_missing.exclude_snps.txt")

    rows: list[dict[str, object]] = []
    source_variants, source_samples = prefix_counts(args.source_prefix)
    rows.append(
        {
            "step": "source",
            "prefix": str(args.source_prefix),
            "variants": source_variants,
            "samples": source_samples,
            "notes": "input PLINK prefix",
        }
    )

    if not plink_path(step1, ".bed").exists():
        step1_cmd = [
                args.plink_bin,
                "--bfile",
                str(args.source_prefix),
                "--chr",
                "1-22",
                "--geno",
                "0.1",
                "--make-bed",
                "--allow-no-sex",
                "--threads",
                str(args.threads),
                "--out",
                str(step1),
        ]
        if mind_threshold is not None:
            step1_cmd[7:7] = ["--mind", mind_threshold]
        run_cmd(step1_cmd, plink_path(step1, ".stdout.log"))
    variants, samples = prefix_counts(step1)
    rows.append(
        {
            "step": "autosome_geno01_keepSamples" if mind_threshold is None else "autosome_geno01_mind",
            "prefix": str(step1),
            "variants": variants,
            "samples": samples,
            "notes": "--chr 1-22 --geno 0.1" + (f" --mind {mind_threshold}" if mind_threshold is not None else " (sample QC disabled)"),
        }
    )

    if not plink_path(step2, ".bed").exists():
        run_cmd(
            [
                args.plink_bin,
                "--bfile",
                str(step1),
                "--geno",
                "0.0",
                "--maf",
                "0.01",
                "--hwe",
                "1e-5",
                "midp",
                "include-nonctrl",
                "--make-bed",
                "--allow-no-sex",
                "--threads",
                str(args.threads),
                "--out",
                str(step2),
            ],
            plink_path(step2, ".stdout.log"),
        )
    variants, samples = prefix_counts(step2)
    rows.append(
        {
            "step": "strict_snp_qc",
            "prefix": str(step2),
            "variants": variants,
            "samples": samples,
            "notes": "--geno 0.0 --maf 0.01 --hwe 1e-5 midp include-nonctrl",
        }
    )

    missing_path = plink_path(diff_prefix, ".missing")
    if not missing_path.exists():
        run_cmd(
            [
                args.plink_bin,
                "--bfile",
                str(step2),
                "--test-missing",
                "midp",
                "--allow-no-sex",
                "--threads",
                str(args.threads),
                "--out",
                str(diff_prefix),
            ],
            plink_path(diff_prefix, ".stdout.log"),
        )
    n_diff_missing = parse_diff_missing(missing_path, diff_exclude, args.diff_missing_p)

    if not plink_path(args.out_prefix, ".bed").exists():
        cmd = [
            args.plink_bin,
            "--bfile",
            str(step2),
            "--make-bed",
            "--allow-no-sex",
            "--threads",
            str(args.threads),
            "--out",
            str(args.out_prefix),
        ]
        if n_diff_missing:
            cmd[3:3] = ["--exclude", str(diff_exclude)]
        run_cmd(cmd, plink_path(args.out_prefix, ".stdout.log"))

    variants, samples = prefix_counts(args.out_prefix)
    rows.append(
        {
            "step": "final",
            "prefix": str(args.out_prefix),
            "variants": variants,
            "samples": samples,
            "notes": f"excluded {n_diff_missing} SNPs with differential missingness P <= {args.diff_missing_p:g}",
        }
    )
    report_path = args.out_prefix.with_name(args.out_prefix.name + ".qc_report.tsv")
    write_report(report_path, rows)
    print(f"Wrote QC report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
