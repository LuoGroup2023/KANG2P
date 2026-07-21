#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash run_classification_benchmarks.sh \
#     --features pkl:/path/genotype.pkl \
#     --split-dir /path/cv_splits_5fold \
#     --output-dir outputs/traditional_ml/als_g \
#     --prefix als_g

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TOTAL_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
GRID_JOBS="${ML_GRID_JOBS:-1}"
RF_JOBS="${ML_RF_JOBS:-1}"

if (( GRID_JOBS < 1 || RF_JOBS < 1 )); then
  echo "ML_GRID_JOBS and ML_RF_JOBS must be positive" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "Traditional ML nested CV: grid_jobs=${GRID_JOBS}, rf_jobs=${RF_JOBS}, host_cpus=${TOTAL_CPUS}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/classification_nested_cv.py" \
  --grid-jobs "${GRID_JOBS}" \
  --rf-jobs "${RF_JOBS}" \
  "$@"
