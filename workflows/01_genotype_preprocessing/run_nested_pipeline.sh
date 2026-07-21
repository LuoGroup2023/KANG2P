#!/usr/bin/env bash
set -euo pipefail

# Fold-safe ALS genotype preprocessing.
#
# Required environment variables:
#   SOURCE_PREFIX  Input PLINK prefix (.bed/.bim/.fam)
#   SPLIT_DIR      outer_fold_{1..5}_{train,test}_IDs.txt directory
#   ANNOVAR_DIR    ANNOVAR directory containing annotate_variation.pl and humandb/
#
# Optional:
#   OUT_ROOT, PLINK_BIN, PYTHON_BIN, THREADS, PVALUE_THRESHOLD,
#   MIND_THRESHOLD, FOLDS, MAX_GWAS_SNPS, MAX_SNPS_PER_GENE.
#
# Usage:
#   SOURCE_PREFIX=/path/cohort SPLIT_DIR=/path/folds ANNOVAR_DIR=/opt/annovar \
#     bash run_nested_pipeline.sh all
#   bash run_nested_pipeline.sh qc
#   bash run_nested_pipeline.sh gene-pca --skip-existing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="${1:-all}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
PLINK_BIN="${PLINK_BIN:-plink}"
SOURCE_PREFIX="${SOURCE_PREFIX:-${REPO_ROOT}/data/ALS/02_merged_all_QCedSNPs}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/ALS/cv_splits_5fold}"
ANNOVAR_DIR="${ANNOVAR_DIR:-${REPO_ROOT}/tools/annovar}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/ALS/genotype_preprocessing}"
QC_PREFIX="${QC_PREFIX:-${OUT_ROOT}/paper_qc/03_merged_all_paperQC_autosome}"
GENE_PCA_OUT="${GENE_PCA_OUT:-${OUT_ROOT}/nested_gene_pca}"
THREADS="${THREADS:-8}"
PVALUE_THRESHOLD="${PVALUE_THRESHOLD:-0.05}"
MIND_THRESHOLD="${MIND_THRESHOLD:-none}"
FOLDS="${FOLDS:-all}"
MAX_GWAS_SNPS="${MAX_GWAS_SNPS:-0}"
MAX_SNPS_PER_GENE="${MAX_SNPS_PER_GENE:-0}"
GENES_PER_BATCH="${GENES_PER_BATCH:-1000}"
SNPS_PER_BATCH="${SNPS_PER_BATCH:-25000}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

require_command() {
  if [[ "$1" == */* ]]; then
    [[ -x "$1" ]] || { echo "Not executable: $1" >&2; exit 2; }
  else
    command -v "$1" >/dev/null 2>&1 || { echo "Command not found: $1" >&2; exit 2; }
  fi
}

run_qc() {
  require_command "${PLINK_BIN}"
  require_file "${SOURCE_PREFIX}.bed"
  require_file "${SOURCE_PREFIX}.bim"
  require_file "${SOURCE_PREFIX}.fam"
  mkdir -p "$(dirname "${QC_PREFIX}")"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_paper_qc.py" \
    --source-prefix "${SOURCE_PREFIX}" \
    --out-prefix "${QC_PREFIX}" \
    --plink-bin "${PLINK_BIN}" \
    --threads "${THREADS}" \
    --mind-threshold "${MIND_THRESHOLD}"
}

run_gene_pca() {
  local raw_prefix="$1"
  shift
  require_command "${PLINK_BIN}"
  require_file "${raw_prefix}.bed"
  require_file "${raw_prefix}.bim"
  require_file "${raw_prefix}.fam"
  require_file "${ANNOVAR_DIR}/annotate_variation.pl"
  require_file "${ANNOVAR_DIR}/humandb/hg19_refGene.txt"
  [[ -d "${SPLIT_DIR}" ]] || { echo "Missing split directory: ${SPLIT_DIR}" >&2; exit 2; }

  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_nested_gwas_gene_pca.py" \
    --raw-prefix "${raw_prefix}" \
    --fold-index-fam-prefix "${SOURCE_PREFIX}" \
    --split-dir "${SPLIT_DIR}" \
    --out-dir "${GENE_PCA_OUT}" \
    --annovar-dir "${ANNOVAR_DIR}" \
    --plink-bin "${PLINK_BIN}" \
    --threads "${THREADS}" \
    --folds "${FOLDS}" \
    --pvalue-threshold "${PVALUE_THRESHOLD}" \
    --max-gwas-snps "${MAX_GWAS_SNPS}" \
    --max-snps-per-gene "${MAX_SNPS_PER_GENE}" \
    --genes-per-batch "${GENES_PER_BATCH}" \
    --snps-per-batch "${SNPS_PER_BATCH}" \
    "$@"
}

case "${MODE}" in
  qc)
    run_qc
    ;;
  gene-pca)
    run_gene_pca "${RAW_PREFIX:-${QC_PREFIX}}" "$@"
    ;;
  all)
    run_qc
    run_gene_pca "${QC_PREFIX}" "$@"
    ;;
  *)
    echo "Usage: $0 {all|qc|gene-pca} [run_nested_gwas_gene_pca.py options]" >&2
    exit 2
    ;;
esac
