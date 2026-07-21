#!/usr/bin/env bash
set -euo pipefail

# Run DiseaseCapsule and traditional-ML PP comparisons from one aligned table.
# Required: PP_TABLE, LABEL_PKL, SPLIT_DIR. Optional: OUT_ROOT, PYTHON_BIN.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${PP_TABLE:?Set PP_TABLE to the prepared predicted-proteome TSV}"
: "${LABEL_PKL:?Set LABEL_PKL to the cohort label PKL}"
: "${SPLIT_DIR:?Set SPLIT_DIR to the outer-fold index directory}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/predicted_proteome}"

PYTHON_BIN="${PYTHON_BIN}" \
  bash "${REPO_ROOT}/workflows/05_comparison_experiments/run_disease_baselines.sh" \
    --features "tsv:${PP_TABLE}" \
    --labels "pkl:${LABEL_PKL}" \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT_ROOT}" \
    --prefix pp \
    "$@"
