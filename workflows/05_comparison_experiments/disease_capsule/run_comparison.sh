#!/usr/bin/env bash
set -euo pipefail

# Thin reproducible launcher for G/PE/PP DiseaseCapsule comparisons.
# Repeat --features to concatenate modalities in their already aligned order.
#
# Example (AD predicted proteome):
#   bash run_comparison.sh \
#     --features tsv:/path/predicted_proteome.tsv \
#     --labels pkl:/path/labels.pkl \
#     --split-dir /path/cv_splits_5fold_AD \
#     --output-dir outputs/disease_capsule/ad_pp \
#     --prefix ad_pp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/capsnet_nested_cv.py" \
  --device "${CAPSNET_DEVICE:-auto}" \
  --n-trials "${CAPSNET_TRIALS:-15}" \
  --inner-folds "${CAPSNET_INNER_FOLDS:-3}" \
  --tune-epochs "${CAPSNET_TUNE_EPOCHS:-15}" \
  --final-epochs "${CAPSNET_FINAL_EPOCHS:-30}" \
  "$@"
