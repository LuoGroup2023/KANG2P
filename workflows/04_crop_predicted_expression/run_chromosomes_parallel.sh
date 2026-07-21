#!/usr/bin/env bash
set -uo pipefail

# Parallel chromosome launcher for cis_elastic_net_expression.R.
# Required: DATA_DIR, GFF_FILE, GENE_EXP, PREDICT_FAM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
PLINK_BIN="${PLINK_BIN:-plink}"
: "${DATA_DIR:?Set DATA_DIR to the crop genotype directory}"
: "${GFF_FILE:?Set GFF_FILE to the crop GFF3 annotation}"
: "${GENE_EXP:?Set GENE_EXP to the observed expression matrix}"
: "${PREDICT_FAM:?Set PREDICT_FAM to the target-sample FAM file}"

CHROMOSOMES="${CHROMOSOMES:-1-12}"
if [[ -z "${PLINK_PREFIX_TEMPLATE:-}" ]]; then
  PLINK_PREFIX_TEMPLATE="${DATA_DIR}/chr{chr}_impute"
fi
CHR_PREFIX="${CHR_PREFIX:-Chr}"
GENE_EXP_SEP="${GENE_EXP_SEP:-tab}"
FAM_ID_COLUMN="${FAM_ID_COLUMN:-1}"
MAX_JOBS="${MAX_JOBS:-4}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/crop_predicted_expression/$(date +%Y%m%d_%H%M%S)}"
MODEL_SCRIPT="${MODEL_SCRIPT:-${SCRIPT_DIR}/cis_elastic_net_expression.R}"

expand_range_list() {
  local token start end value
  local text="${1//,/ }"
  for token in ${text}; do
    if [[ "${token}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      for ((value = start; value <= end; value++)); do echo "${value}"; done
    else
      echo "${token}"
    fi
  done
}

render_template() {
  local template="$1"
  local placeholder="$2"
  local replacement="$3"
  local before after
  [[ "${template}" == *"${placeholder}"* ]] || {
    echo "Template does not contain ${placeholder}: ${template}" >&2
    return 2
  }
  before="${template%%"${placeholder}"*}"
  after="${template#*"${placeholder}"}"
  printf '%s%s%s' "${before}" "${replacement}" "${after}"
}

require_file() {
  [[ -s "$1" ]] || { echo "Missing or empty required file: $1" >&2; exit 2; }
}
require_command() {
  if [[ "$1" == */* ]]; then
    [[ -x "$1" ]] || { echo "Not executable: $1" >&2; exit 2; }
  else
    command -v "$1" >/dev/null 2>&1 || { echo "Command not found: $1" >&2; exit 2; }
  fi
}

[[ "${PLINK_PREFIX_TEMPLATE}" == *'{chr}'* ]] || {
  echo "PLINK_PREFIX_TEMPLATE must contain {chr}" >&2
  exit 2
}
[[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_JOBS must be positive" >&2; exit 2; }
require_command "${RSCRIPT_BIN}"
require_command "${PLINK_BIN}"
require_file "${MODEL_SCRIPT}"
require_file "${GFF_FILE}"
require_file "${GENE_EXP}"
require_file "${PREDICT_FAM}"
mapfile -t CHR_LIST < <(expand_range_list "${CHROMOSOMES}")

mkdir -p "${OUT_ROOT}/logs"
printf 'Chromosome\tPID\tPlinkPrefix\tOutputDir\tLog\n' > "${OUT_ROOT}/jobs.tsv"

active=0
failures=0
for chromosome in "${CHR_LIST[@]}"; do
  while ((active >= MAX_JOBS)); do
    if ! wait -n; then failures=$((failures + 1)); fi
    active=$((active - 1))
  done

  plink_prefix="$(render_template "${PLINK_PREFIX_TEMPLATE}" '{chr}' "${chromosome}")"
  require_file "${plink_prefix}.bed"
  require_file "${plink_prefix}.bim"
  require_file "${plink_prefix}.fam"
  chromosome_out="${OUT_ROOT}/chr${chromosome}"
  log_file="${OUT_ROOT}/logs/chr${chromosome}.log"
  mkdir -p "${chromosome_out}"

  echo "[SNP2Expression] launching chromosome ${chromosome}"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  "${RSCRIPT_BIN}" "${MODEL_SCRIPT}" \
    --plink_prefix "${plink_prefix}" \
    --chr "${chromosome}" \
    --chr_prefix "${CHR_PREFIX}" \
    --output_dir "${chromosome_out}" \
    --gff "${GFF_FILE}" \
    --gene_exp "${GENE_EXP}" \
    --gene_exp_sep "${GENE_EXP_SEP}" \
    --predict_fam "${PREDICT_FAM}" \
    --plink_bin "${PLINK_BIN}" \
    --fam_id_column "${FAM_ID_COLUMN}" \
    "$@" > "${log_file}" 2>&1 &

  pid=$!
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${chromosome}" "${pid}" "${plink_prefix}" "${chromosome_out}" "${log_file}" \
    >> "${OUT_ROOT}/jobs.tsv"
  active=$((active + 1))
done

while ((active > 0)); do
  if ! wait -n; then failures=$((failures + 1)); fi
  active=$((active - 1))
done

if ((failures > 0)); then
  echo "Finished with ${failures} failed chromosome job(s): ${OUT_ROOT}" >&2
  exit 1
fi
echo "All chromosome jobs finished: ${OUT_ROOT}"
