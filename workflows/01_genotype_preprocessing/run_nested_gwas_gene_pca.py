#!/usr/bin/env python3
"""Run fold-specific GWAS and Gene-PCA from the original PLINK data.

This script follows the DiseaseCapsule preprocessing shape, but repeats the
GWAS/SNP-selection/Gene-PCA steps inside each outer fold. For each fold:

1. Use unique outer-train samples for PLINK GWAS.
2. Select GWAS SNPs by p-value.
3. Annotate selected SNPs to genes with ANNOVAR hg19 refGene.
4. Extract selected SNP dosages from the original PLINK bed/bim/fam.
5. Fit PCA per gene on unique outer-train samples, transform all samples, and
   export train/test matrices exactly in the order listed by the fold ID files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import pickle
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


META_COLS = ("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_folds(text: str) -> list[int]:
    if text.strip().lower() == "all":
        return [1, 2, 3, 4, 5]
    folds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))
    return folds


def unique_in_order(values: np.ndarray) -> np.ndarray:
    _, first_pos = np.unique(values, return_index=True)
    return values[np.sort(first_pos)]


def run_cmd(cmd: list[str], log_path: Path | None = None) -> None:
    print("+ " + " ".join(map(str, cmd)), flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    with log_path.open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def read_fam(fam_path: Path) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    rows: list[list[str]] = []
    sample_ids: list[str] = []
    labels: list[int] = []
    with fam_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"Malformed FAM line: {line!r}")
            rows.append(fields)
            sample_ids.append(fields[1])
            pheno = int(float(fields[5]))
            if pheno == 1:
                labels.append(0)
            elif pheno == 2:
                labels.append(1)
            else:
                labels.append(-1)
    return rows, np.asarray(sample_ids, dtype=object), np.asarray(labels, dtype=np.int64)


def sample_key(row: list[str]) -> tuple[str, str]:
    return row[0], row[1]


def read_fold_ids(split_dir: Path, fold: int) -> tuple[np.ndarray, np.ndarray]:
    train_path = split_dir / f"outer_fold_{fold}_train_IDs.txt"
    test_path = split_dir / f"outer_fold_{fold}_test_IDs.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing fold files for fold {fold} in {split_dir}")
    train_idx = np.array([int(x) for x in train_path.read_text().split()], dtype=np.int64)
    test_idx = np.array([int(x) for x in test_path.read_text().split()], dtype=np.int64)
    return train_idx, test_idx


def validate_fold_indices(indices: Iterable[np.ndarray], n_samples: int) -> None:
    for arr in indices:
        if arr.size == 0:
            raise ValueError("Fold index file is empty.")
        if int(arr.min()) < 0 or int(arr.max()) >= n_samples:
            raise ValueError(
                f"Fold index out of bounds for n_samples={n_samples}: "
                f"min={arr.min()}, max={arr.max()}"
            )


def map_fold_indices_to_current_fam(
    fold_indices: np.ndarray,
    fold_fam_rows: list[list[str]],
    current_fam_rows: list[list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Map split-file row indices from their source FAM to the current FAM.

    Existing nested-CV split files are row indices into the pre-QC FAM. Strict
    sample QC may remove people, so we map by FID/IID and drop removed samples
    while preserving the split order and any upsampling duplicates.
    """
    current_pos_by_key = {sample_key(row): i for i, row in enumerate(current_fam_rows)}
    mapped: list[int] = []
    dropped: list[int] = []
    for idx in fold_indices:
        key = sample_key(fold_fam_rows[int(idx)])
        pos = current_pos_by_key.get(key)
        if pos is None:
            dropped.append(int(idx))
        else:
            mapped.append(pos)
    return np.asarray(mapped, dtype=np.int64), np.asarray(dropped, dtype=np.int64)


def write_plink_keep(path: Path, fam_rows: list[list[str]], idx: np.ndarray) -> None:
    with path.open("w") as fh:
        for i in idx:
            row = fam_rows[int(i)]
            fh.write(f"{row[0]}\t{row[1]}\n")


def parse_assoc(assoc_path: Path, pvalue_threshold: float) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    reader = pd.read_csv(
        assoc_path,
        sep=r"\s+",
        engine="c",
        chunksize=1_000_000,
        na_values=["NA", "nan", "NaN"],
    )
    for chunk in reader:
        if "SNP" not in chunk.columns or "P" not in chunk.columns:
            raise ValueError(f"{assoc_path} does not look like a PLINK --assoc file.")
        pvals = pd.to_numeric(chunk["P"], errors="coerce")
        keep = pvals.notna() & (pvals <= pvalue_threshold)
        if keep.any():
            part = chunk.loc[keep, ["CHR", "SNP", "BP", "P"]].copy()
            part["P"] = pvals.loc[keep].astype(float).to_numpy()
            selected.append(part)
    if not selected:
        return pd.DataFrame(columns=["CHR", "SNP", "BP", "P"])
    out = pd.concat(selected, axis=0, ignore_index=True)
    out = out.sort_values(["P", "CHR", "BP", "SNP"], kind="mergesort")
    out = out.drop_duplicates("SNP", keep="first")
    return out


def load_bim_records(bim_path: Path, snps: set[str]) -> dict[str, tuple[str, int, str, str]]:
    records: dict[str, tuple[str, int, str, str]] = {}
    with bim_path.open() as fh:
        for line in fh:
            chrom, snp, _cm, pos, a1, a2 = line.split()[:6]
            if snp in snps:
                records[snp] = (chrom, int(pos), a1, a2)
    missing = snps - records.keys()
    if missing:
        raise ValueError(f"{len(missing)} selected SNPs are missing from BIM; first={next(iter(missing))}")
    return records


def write_avinput(path: Path, assoc: pd.DataFrame, bim_records: dict[str, tuple[str, int, str, str]]) -> None:
    with path.open("w") as fh:
        for snp in assoc["SNP"]:
            chrom, pos, a1, a2 = bim_records[str(snp)]
            fh.write(f"{chrom}\t{pos}\t{pos}\t{a1}\t{a2}\t{snp}\n")


def clean_gene_name(gene: str) -> str:
    gene = re.sub(r"\([^)]*\)", "", gene).strip()
    gene = gene.replace("/", "_")
    return gene


def nearest_intergenic_gene(gene_field: str) -> str | None:
    match = re.match(r"(.+)\(dist=([^)]*)\),(.+)\(dist=([^)]*)\)$", gene_field)
    if not match:
        return None
    g1, d1, g2, d2 = match.groups()
    candidates: list[tuple[float, str]] = []
    for gene, dist in ((g1, d1), (g2, d2)):
        gene = clean_gene_name(gene)
        if not gene or gene == "NONE" or dist == "NONE":
            continue
        try:
            candidates.append((float(dist), gene))
        except ValueError:
            continue
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1]


def genes_from_annovar_fields(func: str, gene_field: str) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    if func == "intergenic":
        gene = nearest_intergenic_gene(gene_field)
        return [(gene, func)] if gene else []

    funcs = func.split(";")
    gene_field = re.sub(r"\([^)]*\)", "", gene_field)
    gene_groups = gene_field.split(";")

    if len(gene_groups) != len(funcs):
        funcs = [func] * len(gene_groups)

    for sub_func, group in zip(funcs, gene_groups):
        for gene in group.split(","):
            gene = clean_gene_name(gene)
            if gene and gene != "NONE":
                hits.append((gene, sub_func))
    return hits


def parse_variant_function(path: Path, output_map: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open() as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 7:
                continue
            func, gene_field = fields[0], fields[1]
            chrom = fields[2].removeprefix("chr")
            pos = fields[3]
            # ``write_avinput`` stores the original PLINK SNP ID in the last
            # column. ANNOVAR preserves it after the two annotation columns;
            # it may be an rsID and therefore need not contain a colon.
            snp = fields[7] if len(fields) > 7 and fields[7] else f"{chrom}:{pos}"
            for gene, sub_func in genes_from_annovar_fields(func, gene_field):
                rows.append(
                    {
                        "snp": snp,
                        "chrom": chrom,
                        "pos": int(pos),
                        "gene": gene,
                        "function": sub_func,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"ANNOVAR produced no SNP-to-gene assignments: {path}")
    df = df.drop_duplicates(["snp", "gene"]).sort_values(["chrom", "pos", "gene"])
    df.to_csv(output_map, sep="\t", index=False)
    return df


def apply_gene_snp_limits(
    snp_gene: pd.DataFrame,
    assoc: pd.DataFrame,
    max_snps_per_gene: int | None,
) -> dict[str, list[str]]:
    pval = assoc.set_index("SNP")["P"].to_dict()
    pos = assoc.set_index("SNP")["BP"].to_dict()
    gene_to_snps: dict[str, list[str]] = {}
    for gene, sub in snp_gene.groupby("gene", sort=True):
        snps = list(dict.fromkeys(sub["snp"].astype(str).tolist()))
        snps = sorted(snps, key=lambda snp: (float(pval.get(snp, math.inf)), int(pos.get(snp, 0)), snp))
        if max_snps_per_gene:
            snps = snps[:max_snps_per_gene]
        if snps:
            gene_to_snps[str(gene)] = snps
    return gene_to_snps


def raw_column_to_snp(column: str, selected_snps: set[str]) -> str | None:
    if column in selected_snps:
        return column
    if "_" in column:
        base = column.rsplit("_", 1)[0]
        if base in selected_snps:
            return base
    return None


def pca_components_for_gene(n_snps: int, n_train: int) -> int:
    if n_snps > 20:
        requested = 8
    elif n_snps > 4:
        requested = 4
    else:
        requested = 1
    return max(1, min(requested, n_snps, n_train))


def build_gene_pca_features(
    plink_bin: str,
    source_prefix: Path,
    gene_to_snps: dict[str, list[str]],
    train_fit_idx: np.ndarray,
    all_labels: np.ndarray,
    output_prefix: Path,
    batch_dir: Path,
    threads: int,
    genes_per_batch: int,
    snps_per_batch: int,
    keep_plink_raw: bool,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    embeddings: list[np.ndarray] = []
    feature_names: list[str] = []
    gene_rows: list[dict[str, object]] = []
    train_fit_idx = np.asarray(train_fit_idx, dtype=np.int64)
    batch_dir.mkdir(parents=True, exist_ok=True)

    gene_items = sorted(gene_to_snps.items())
    batches: list[list[tuple[str, list[str]]]] = []
    current: list[tuple[str, list[str]]] = []
    current_snps: set[str] = set()
    for gene, snps in gene_items:
        snp_set = set(snps)
        would_exceed_gene_count = genes_per_batch > 0 and len(current) >= genes_per_batch
        would_exceed_snp_count = snps_per_batch > 0 and len(current_snps | snp_set) > snps_per_batch
        if current and (would_exceed_gene_count or would_exceed_snp_count):
            batches.append(current)
            current = []
            current_snps = set()
        current.append((gene, snps))
        current_snps.update(snp_set)
    if current:
        batches.append(current)

    n_samples: int | None = None
    print(
        f"Running Gene-PCA in {len(batches)} PLINK dosage batches "
        f"(genes_per_batch={genes_per_batch}, snps_per_batch={snps_per_batch})",
        flush=True,
    )

    for batch_i, batch in enumerate(batches, start=1):
        batch_snps = sorted({snp for _gene, snps in batch for snp in snps})
        batch_extract = batch_dir / f"batch_{batch_i:04d}.snps.txt"
        batch_prefix = batch_dir / f"batch_{batch_i:04d}"
        batch_raw = batch_prefix.with_suffix(".raw")
        with batch_extract.open("w") as fh:
            fh.write("\n".join(batch_snps) + "\n")

        if not batch_raw.exists():
            run_cmd(
                [
                    plink_bin,
                    "--bfile",
                    str(source_prefix),
                    "--extract",
                    str(batch_extract),
                    "--recode",
                    "A",
                    "--allow-no-sex",
                    "--threads",
                    str(threads),
                    "--out",
                    str(batch_prefix),
                ],
                log_path=batch_prefix.with_suffix(".plink.log"),
            )

        print(
            f"Loading dosage batch {batch_i}/{len(batches)}: "
            f"genes={len(batch)}, snps={len(batch_snps)}",
            flush=True,
        )
        raw_df = pd.read_csv(batch_raw, sep=r"\s+", engine="c", na_values=["NA"])
        if n_samples is None:
            n_samples = raw_df.shape[0]
        elif raw_df.shape[0] != n_samples:
            raise ValueError(f"Batch {batch_i} sample count changed: {raw_df.shape[0]} vs {n_samples}")

        raw_col_by_snp: dict[str, str] = {}
        batch_snp_set = set(batch_snps)
        for col in raw_df.columns:
            if col in META_COLS:
                continue
            snp = raw_column_to_snp(str(col), batch_snp_set)
            if snp is not None:
                raw_col_by_snp[snp] = str(col)

        for gene_pos, (gene, gene_snps) in enumerate(batch, start=1):
            if gene_pos == 1 or gene_pos % 50 == 0 or gene_pos == len(batch):
                print(
                    f"  batch {batch_i}/{len(batches)} PCA gene {gene_pos}/{len(batch)}",
                    flush=True,
                )
            snps = [snp for snp in gene_snps if snp in raw_col_by_snp]
            if not snps:
                continue
            cols = [raw_col_by_snp[snp] for snp in snps]
            x_gene = raw_df[cols].to_numpy(dtype=np.float32, copy=True)
            # Impute from the unique outer-training samples only. Using a
            # sentinel value (or statistics from all samples) would distort
            # PCA and leak information from the held-out outer fold.
            train_means = np.nanmean(x_gene[train_fit_idx], axis=0)
            train_means = np.nan_to_num(train_means, nan=0.0).astype(np.float32)
            missing_rows, missing_cols = np.where(~np.isfinite(x_gene))
            if missing_rows.size:
                x_gene[missing_rows, missing_cols] = train_means[missing_cols]
            n_comp = pca_components_for_gene(x_gene.shape[1], len(train_fit_idx))

            pca = PCA(n_components=n_comp)
            pca.fit(x_gene[train_fit_idx])
            emb = pca.transform(x_gene).astype(np.float32, copy=False)
            embeddings.append(emb)

            chrom = str(snps[0]).split(":", 1)[0]
            feature_names.extend([f"chr{chrom}:{gene}:{i}" for i in range(n_comp)])
            gene_rows.append(
                {
                    "gene": gene,
                    "chrom": chrom,
                    "n_snps": len(snps),
                    "n_components": n_comp,
                    "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
                }
            )

        if not keep_plink_raw and batch_raw.exists():
            batch_raw.unlink()
        del raw_df
        gc.collect()

    if not embeddings:
        raise ValueError("No gene-level PCA embeddings were produced.")

    x_pca = np.concatenate(embeddings, axis=1).astype(np.float32, copy=False)
    gene_summary = pd.DataFrame(gene_rows)
    gene_summary.to_csv(output_prefix.with_suffix(".gene_pca_summary.tsv"), sep="\t", index=False)
    with output_prefix.with_suffix(".feature_names.txt").open("w") as fh:
        fh.write("\n".join(feature_names) + "\n")
    print(
        f"Gene-PCA matrix shape: samples={n_samples}, features={x_pca.shape[1]}, "
        f"genes={len(gene_summary)}",
        flush=True,
    )
    if n_samples is None or len(all_labels) != n_samples:
        raise ValueError(f"Label count ({len(all_labels)}) != dosage rows ({n_samples})")
    return x_pca, feature_names, gene_summary


def count_labels(y: np.ndarray) -> tuple[int, int]:
    return int(np.sum(y == 0)), int(np.sum(y == 1))


def write_dimension_report(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "fold",
        "output_pkl",
        "source_samples",
        "n_selected_snps",
        "n_annotated_snps",
        "n_genes",
        "n_features",
        "train_source_rows",
        "train_rows",
        "train_dropped_rows",
        "train_unique_rows",
        "train_label0",
        "train_label1",
        "test_source_rows",
        "test_rows",
        "test_dropped_rows",
        "test_unique_rows",
        "test_label0",
        "test_label1",
    ]
    # Preserve folds skipped by --skip-existing and replace only folds that
    # were recomputed during this invocation.
    by_fold: dict[int, dict[str, object]] = {}
    if path.exists():
        with path.open(newline="") as existing_fh:
            for existing in csv.DictReader(existing_fh, delimiter="\t"):
                by_fold[int(existing["fold"])] = existing
    for row in rows:
        by_fold[int(row["fold"])] = row

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(by_fold[fold] for fold in sorted(by_fold))


def build_arg_parser() -> argparse.ArgumentParser:
    root = project_root()
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Nested-CV fold-specific GWAS + Gene-PCA from PLINK raw data.")
    parser.add_argument("--raw-prefix", type=Path, default=root / "data/ALS/02_merged_all_QCedSNPs")
    parser.add_argument(
        "--fold-index-fam-prefix",
        type=Path,
        default=None,
        help="PLINK prefix whose .fam row numbers are used by the nested-CV split files. Defaults to --raw-prefix.",
    )
    parser.add_argument("--split-dir", type=Path, default=root / "data/ALS/cv_splits_5fold")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/ALS/nested_gene_pca")
    parser.add_argument("--annovar-dir", type=Path, default=root / "tools/annovar")
    parser.add_argument("--plink-bin", default=os.environ.get("PLINK_BIN", "plink"))
    parser.add_argument("--folds", default="all")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--pvalue-threshold", type=float, default=0.05)
    parser.add_argument("--max-gwas-snps", type=int, default=0, help="0 means keep all SNPs passing the p-value threshold.")
    parser.add_argument("--max-snps-per-gene", type=int, default=0, help="0 means no cap; original script had this cap disabled.")
    parser.add_argument("--genes-per-batch", type=int, default=1000, help="Maximum genes per dosage extraction batch.")
    parser.add_argument("--snps-per-batch", type=int, default=25000, help="Maximum unique SNPs per dosage extraction batch.")
    parser.add_argument("--keep-plink-raw", action="store_true", help="Keep batch PLINK .raw dosage intermediates.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip folds with existing output pkl.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    folds = parse_folds(args.folds)

    fam_path = Path(str(args.raw_prefix) + ".fam")
    bim_path = Path(str(args.raw_prefix) + ".bim")
    bed_path = Path(str(args.raw_prefix) + ".bed")
    for path in (fam_path, bim_path, bed_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not Path(args.plink_bin).exists() and shutil.which(args.plink_bin) is None:
        raise FileNotFoundError(f"Cannot find PLINK executable: {args.plink_bin}")

    annovar = args.annovar_dir
    annotate_variation = annovar / "annotate_variation.pl"
    humandb = annovar / "humandb"
    if not annotate_variation.exists():
        raise FileNotFoundError(f"Cannot find ANNOVAR annotate_variation.pl: {annotate_variation}")
    if not (humandb / "hg19_refGene.txt").exists():
        raise FileNotFoundError(f"Cannot find hg19_refGene.txt under {humandb}")

    fam_rows, sample_ids, labels = read_fam(fam_path)
    fold_index_fam_path = (
        Path(str(args.fold_index_fam_prefix) + ".fam")
        if args.fold_index_fam_prefix is not None
        else fam_path
    )
    if not fold_index_fam_path.exists():
        raise FileNotFoundError(fold_index_fam_path)
    fold_fam_rows, _fold_sample_ids, _fold_labels = (
        (fam_rows, sample_ids, labels)
        if fold_index_fam_path == fam_path
        else read_fam(fold_index_fam_path)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    max_snps_per_gene = args.max_snps_per_gene if args.max_snps_per_gene > 0 else None
    dimension_rows: list[dict[str, object]] = []

    for fold in folds:
        fold_dir = args.out_dir / f"outer_fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        output_pkl = fold_dir / f"outer_fold_{fold}_gene_pca.pkl"
        if args.skip_existing and output_pkl.exists():
            print(f"fold {fold}: output exists, skipping: {output_pkl}", flush=True)
            continue

        source_train_idx, source_test_idx = read_fold_ids(args.split_dir, fold)
        validate_fold_indices((source_train_idx, source_test_idx), len(fold_fam_rows))
        train_idx, dropped_train_idx = map_fold_indices_to_current_fam(
            source_train_idx,
            fold_fam_rows,
            fam_rows,
        )
        test_idx, dropped_test_idx = map_fold_indices_to_current_fam(
            source_test_idx,
            fold_fam_rows,
            fam_rows,
        )
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError(
                f"fold {fold}: no samples remain after mapping split indices to {fam_path}"
            )
        if dropped_train_idx.size or dropped_test_idx.size:
            print(
                f"fold {fold}: dropped {len(dropped_train_idx)} train rows and "
                f"{len(dropped_test_idx)} test rows absent from QC FAM",
                flush=True,
            )
        validate_fold_indices((train_idx, test_idx), len(fam_rows))
        train_fit_idx = unique_in_order(train_idx)

        keep_path = fold_dir / f"outer_fold_{fold}_train_unique.keep"
        write_plink_keep(keep_path, fam_rows, train_fit_idx)

        gwas_prefix = fold_dir / "gwas" / f"outer_fold_{fold}"
        gwas_prefix.parent.mkdir(parents=True, exist_ok=True)
        assoc_path = gwas_prefix.with_suffix(".assoc")
        if not assoc_path.exists():
            run_cmd(
                [
                    args.plink_bin,
                    "--bfile",
                    str(args.raw_prefix),
                    "--keep",
                    str(keep_path),
                    "--assoc",
                    "--allow-no-sex",
                    "--threads",
                    str(args.threads),
                    "--out",
                    str(gwas_prefix),
                ],
                log_path=gwas_prefix.with_suffix(".plink.log"),
            )

        selected_assoc_path = fold_dir / f"outer_fold_{fold}_gwas_p{args.pvalue_threshold:g}.snps.tsv"
        if selected_assoc_path.exists() and args.max_gwas_snps == 0:
            print(f"fold {fold}: reusing selected GWAS SNP table {selected_assoc_path}", flush=True)
            selected_assoc = pd.read_csv(selected_assoc_path, sep="\t")
        else:
            selected_assoc = parse_assoc(assoc_path, args.pvalue_threshold)
            if selected_assoc.empty:
                raise ValueError(f"fold {fold}: no SNPs pass p <= {args.pvalue_threshold}")
            if args.max_gwas_snps > 0:
                selected_assoc = selected_assoc.head(args.max_gwas_snps).copy()
            selected_assoc.to_csv(selected_assoc_path, sep="\t", index=False)
        selected_snps = set(selected_assoc["SNP"].astype(str))

        bim_records = load_bim_records(bim_path, selected_snps)
        annovar_dir = fold_dir / "annovar"
        annovar_dir.mkdir(exist_ok=True)
        avinput = annovar_dir / f"outer_fold_{fold}.avinput"
        write_avinput(avinput, selected_assoc, bim_records)
        annovar_prefix = annovar_dir / f"outer_fold_{fold}"
        variant_function = annovar_prefix.with_suffix(".variant_function")
        if not variant_function.exists():
            run_cmd(
                [
                    str(annotate_variation),
                    "-out",
                    str(annovar_prefix),
                    "-build",
                    "hg19",
                    str(avinput),
                    str(humandb),
                ],
                log_path=annovar_prefix.with_suffix(".annovar.log"),
            )

        snp_gene_map_path = fold_dir / f"outer_fold_{fold}_snp_gene_map.tsv"
        snp_gene = parse_variant_function(variant_function, snp_gene_map_path)
        gene_to_snps = apply_gene_snp_limits(snp_gene, selected_assoc, max_snps_per_gene)

        extract_path = fold_dir / f"outer_fold_{fold}_selected_snps.txt"
        with extract_path.open("w") as fh:
            fh.write("\n".join(sorted({snp for snps in gene_to_snps.values() for snp in snps})) + "\n")

        pca_prefix = fold_dir / f"outer_fold_{fold}_gene_pca_all_samples"
        x_all, feature_names, gene_summary = build_gene_pca_features(
            plink_bin=args.plink_bin,
            source_prefix=args.raw_prefix,
            gene_to_snps=gene_to_snps,
            train_fit_idx=train_fit_idx,
            all_labels=labels,
            output_prefix=pca_prefix,
            batch_dir=fold_dir / f"plink_gene_batches_g{args.genes_per_batch}_s{args.snps_per_batch}",
            threads=args.threads,
            genes_per_batch=args.genes_per_batch,
            snps_per_batch=args.snps_per_batch,
            keep_plink_raw=args.keep_plink_raw,
        )

        x_train = x_all[train_idx]
        y_train = labels[train_idx]
        x_test = x_all[test_idx]
        y_test = labels[test_idx]
        bundle = {
            "fold": fold,
            "raw_prefix": str(args.raw_prefix),
            "split_dir": str(args.split_dir),
            "fold_index_fam": str(fold_index_fam_path),
            "gwas_assoc": str(assoc_path),
            "selected_assoc": str(selected_assoc_path),
            "snp_gene_map": str(snp_gene_map_path),
            "pvalue_threshold": args.pvalue_threshold,
            "max_snps_per_gene": max_snps_per_gene,
            "feature_names": feature_names,
            "source_train_idx": source_train_idx,
            "source_test_idx": source_test_idx,
            "dropped_source_train_idx": dropped_train_idx,
            "dropped_source_test_idx": dropped_test_idx,
            "train_idx": train_idx,
            "train_fit_unique_idx": train_fit_idx,
            "train_sample_ids": sample_ids[train_idx],
            "X_train": x_train,
            "y_train": y_train,
            "test_idx": test_idx,
            "test_sample_ids": sample_ids[test_idx],
            "X_test": x_test,
            "y_test": y_test,
        }
        with output_pkl.open("wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

        train_label0, train_label1 = count_labels(y_train)
        test_label0, test_label1 = count_labels(y_test)
        row = {
            "fold": fold,
            "output_pkl": str(output_pkl),
            "source_samples": len(labels),
            "n_selected_snps": len(selected_snps),
            "n_annotated_snps": int(snp_gene["snp"].nunique()),
            "n_genes": int(gene_summary.shape[0]),
            "n_features": int(x_all.shape[1]),
            "train_source_rows": int(len(source_train_idx)),
            "train_rows": int(len(train_idx)),
            "train_dropped_rows": int(len(dropped_train_idx)),
            "train_unique_rows": int(len(train_fit_idx)),
            "train_label0": train_label0,
            "train_label1": train_label1,
            "test_source_rows": int(len(source_test_idx)),
            "test_rows": int(len(test_idx)),
            "test_dropped_rows": int(len(dropped_test_idx)),
            "test_unique_rows": int(len(np.unique(test_idx))),
            "test_label0": test_label0,
            "test_label1": test_label1,
        }
        dimension_rows.append(row)
        print(
            f"fold {fold}: X_train={x_train.shape}, X_test={x_test.shape}, "
            f"genes={gene_summary.shape[0]}, features={x_all.shape[1]}, output={output_pkl}",
            flush=True,
        )

        del x_all, x_train, x_test, y_train, y_test, bundle
        gc.collect()

    report_path = args.out_dir / "outer_fold_gene_pca_dimensions.tsv"
    write_dimension_report(report_path, dimension_rows)
    print(f"Wrote dimension report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
