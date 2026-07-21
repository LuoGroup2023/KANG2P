#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/plant}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/dualkan_gated_fusion_${RUN_TAG}}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
DEVICE="${DEVICE:-cuda:0}"

N_TRIALS="${N_TRIALS:-12}"
INNER_FOLDS="${INNER_FOLDS:-3}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-3}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
PATIENCE="${PATIENCE:-10}"
TOPK_GENO="${TOPK_GENO:-4096}"
TOPK_EXPR="${TOPK_EXPR:-1024}"
CHUNK_COLS="${CHUNK_COLS:-2048}"
OPT_METRIC="${OPT_METRIC:-composite}"
HEAD_TYPE="${HEAD_TYPE:-fourier}"
FOLDS="${FOLDS:-all}"
SEED="${SEED:-1521024}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_ARRAY[@]}}"

DEFAULT_TASKS=(
  "Maize1404:PH"
  "Maize1404:DTA"
  "Maize1404:KWPE"
  "Maize1404:KNPE"
  "Rice1495:Grain_width"
  "Rice1495:Heading_date"
  "Rice1495:Seed_setting_rate"
  "Rice1495:Yield_per_plant"
  "Rice18K:Plant_height"
  "Rice18K:Culm_length"
  "Rice18K:Grain_length"
  "Rice18K:Grain_width"
  "Rice18K:Grain_yield"
)

if [[ -n "${TASK_LIST:-}" ]]; then
  IFS=',' read -r -a TASKS <<< "${TASK_LIST}"
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

mkdir -p "${OUT_DIR}/logs"

echo "DualKAN caduceus run"
echo "Output: ${OUT_DIR}"
echo "GPUs: ${GPU_LIST}"
echo "Device: ${DEVICE}"
echo "Tasks: ${#TASKS[@]}, max parallel: ${MAX_PARALLEL}"
echo "Trials=${N_TRIALS}, ensemble=${ENSEMBLE_SIZE}, folds=${FOLDS}, opt_metric=${OPT_METRIC}"

launch_task() {
  local task="$1"
  local task_index="$2"
  shift 2
  local dataset="${task%%:*}"
  local trait="${task#*:}"
  local gpu_slot=$((task_index % ${#GPU_ARRAY[@]}))
  local gpu="${GPU_ARRAY[$gpu_slot]}"
  local task_name="${dataset}_${trait}"
  local task_out="${OUT_DIR}/${task_name}"
  local log="${OUT_DIR}/logs/${task_name}.log"

  mkdir -p "${task_out}"
  echo "Launching ${task} on GPU ${gpu}; log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  "${PYTHON}" "${SCRIPT_DIR}/dualkan_gated_fusion.py" \
    --data_root "${DATA_ROOT}" \
    --datasets "${dataset}" \
    --trait "${dataset}:${trait}" \
    --output_dir "${task_out}" \
    --device "${DEVICE}" \
    --gpu_ids 0 \
    --disable_data_parallel \
    --n_trials "${N_TRIALS}" \
    --inner_folds "${INNER_FOLDS}" \
    --ensemble_size "${ENSEMBLE_SIZE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}" \
    --topk_geno "${TOPK_GENO}" \
    --topk_expr "${TOPK_EXPR}" \
    --chunk_cols "${CHUNK_COLS}" \
    --opt_metric "${OPT_METRIC}" \
    --head_type "${HEAD_TYPE}" \
    --folds "${FOLDS}" \
    --seed "$((SEED + task_index * 97))" \
    --num_workers "${NUM_WORKERS}" \
    "$@" > "${log}" 2>&1 &
}

active=0
failures=0
task_index=0

for task in "${TASKS[@]}"; do
  while (( active >= MAX_PARALLEL )); do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active=$((active - 1))
  done
  launch_task "${task}" "${task_index}" "$@"
  active=$((active + 1))
  task_index=$((task_index + 1))
done

while (( active > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  active=$((active - 1))
done

"${PYTHON}" "${SCRIPT_DIR}/aggregate_results.py" "${OUT_DIR}"

if (( failures > 0 )); then
  echo "Finished with ${failures} failed task(s). Check ${OUT_DIR}/logs."
  exit 1
fi

echo "Finished successfully: ${OUT_DIR}"
