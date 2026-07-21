#!/usr/bin/env bash
set -euo pipefail

# Run DiseaseCapsule and traditional classifiers on the same aligned inputs
# and predefined outer folds. Model-specific controls remain environment vars:
# CAPSNET_*, ML_GRID_JOBS, ML_RF_JOBS, and ML_METHODS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FEATURE_ARGS=()
LABEL_ARGS=()
SPLIT_DIR=""
OUTPUT_ROOT=""
PREFIX=""
FOLDS="all"
INNER_FOLDS="3"
SEED="1521024"
KEEP_DUPLICATES=()

usage() {
  cat <<'EOF'
Usage: run_disease_baselines.sh \
  --features pkl:/path/G.pkl [--features tsv:/path/PE.tsv ...] \
  [--labels pkl:/path/labels.pkl] \
  --split-dir /path/cv_splits_5fold \
  --output-root /path/results \
  --prefix experiment_name

Optional common arguments:
  --folds all|1,2,...
  --inner-folds N
  --seed N
  --keep-duplicate-train-rows
EOF
}

while (($# > 0)); do
  case "$1" in
    --features)
      [[ $# -ge 2 ]] || { echo "--features requires a value" >&2; exit 2; }
      FEATURE_ARGS+=(--features "$2")
      shift 2
      ;;
    --labels)
      [[ $# -ge 2 ]] || { echo "--labels requires a value" >&2; exit 2; }
      LABEL_ARGS=(--labels "$2")
      shift 2
      ;;
    --split-dir)
      SPLIT_DIR="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --prefix)
      PREFIX="${2:-}"
      shift 2
      ;;
    --folds)
      FOLDS="${2:-}"
      shift 2
      ;;
    --inner-folds)
      INNER_FOLDS="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --keep-duplicate-train-rows)
      KEEP_DUPLICATES=(--keep-duplicate-train-rows)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

(( ${#FEATURE_ARGS[@]} > 0 )) || { echo "At least one --features input is required" >&2; exit 2; }
[[ -n "${SPLIT_DIR}" ]] || { echo "--split-dir is required" >&2; exit 2; }
[[ -d "${SPLIT_DIR}" ]] || { echo "Split directory not found: ${SPLIT_DIR}" >&2; exit 2; }
[[ -n "${OUTPUT_ROOT}" ]] || { echo "--output-root is required" >&2; exit 2; }
[[ -n "${PREFIX}" ]] || { echo "--prefix is required" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}/disease_capsule" "${OUTPUT_ROOT}/traditional_ml"

COMMON_ARGS=(
  "${FEATURE_ARGS[@]}"
  "${LABEL_ARGS[@]}"
  --split-dir "${SPLIT_DIR}"
  --prefix "${PREFIX}"
  --folds "${FOLDS}"
  --inner-folds "${INNER_FOLDS}"
  --seed "${SEED}"
  "${KEEP_DUPLICATES[@]}"
)

echo "[Comparison] DiseaseCapsule"
PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/disease_capsule/run_comparison.sh" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/disease_capsule"

read -r -a ML_METHOD_LIST <<< "${ML_METHODS:-lr rf svm adaboost}"
echo "[Comparison] Traditional ML: ${ML_METHOD_LIST[*]}"
PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/traditional_ml/run_classification_benchmarks.sh" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/traditional_ml" \
    --methods "${ML_METHOD_LIST[@]}"

echo "Comparison outputs: ${OUTPUT_ROOT}"
