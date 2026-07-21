#!/usr/bin/env bash
set -euo pipefail

# Apply tissue-specific MetaXcan/PrediXcan models to chromosome VCF files.
# Required environment variables:
#   PREDICT_SCRIPT  MetaXcan software/Predict.py
#   MODEL_DIR       directory containing the tissue model databases
#   VCF_TEMPLATE    input pattern containing {chr}, e.g. /secure/ALS/chr{chr}.vcf.gz
# Optional variables are documented in README.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${PREDICT_SCRIPT:?Set PREDICT_SCRIPT to MetaXcan software/Predict.py}"
: "${MODEL_DIR:?Set MODEL_DIR to the tissue-model database directory}"
: "${VCF_TEMPLATE:?Set VCF_TEMPLATE to a path pattern containing {chr}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/human_predicted_expression}"
CHROMOSOMES="${CHROMOSOMES:-1-22}"
if [[ -z "${MODEL_PATTERN:-}" ]]; then
  MODEL_PATTERN='en_{tissue}.db'
fi
MODEL_DB_SNP_KEY="${MODEL_DB_SNP_KEY:-varID}"
VCF_MODE="${VCF_MODE:-genotyped}"
LIFTOVER_CHAIN="${LIFTOVER_CHAIN:-}"
if [[ ! -v MAPPING_TEMPLATE ]]; then
  MAPPING_TEMPLATE='chr{}_{}_{}_{}_b38'
fi
TISSUE_FILE="${TISSUE_FILE:-}"

DEFAULT_TISSUES=(
  Brain_Substantia_nigra
  Brain_Spinal_cord_cervical_c-1
  Brain_Putamen_basal_ganglia
  Brain_Nucleus_accumbens_basal_ganglia
  Brain_Hypothalamus
  Brain_Hippocampus
  Brain_Frontal_Cortex_BA9
  Brain_Cortex
  Brain_Cerebellum
  Brain_Cerebellar_Hemisphere
  Brain_Caudate_basal_ganglia
  Brain_Anterior_cingulate_cortex_BA24
  Brain_Amygdala
)

expand_range_list() {
  local token start end value
  local text="${1//,/ }"
  for token in ${text}; do
    if [[ "${token}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      for ((value = start; value <= end; value++)); do
        echo "${value}"
      done
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

if [[ -n "${TISSUE_FILE}" ]]; then
  [[ -s "${TISSUE_FILE}" ]] || { echo "Missing tissue list: ${TISSUE_FILE}" >&2; exit 2; }
  mapfile -t TISSUES < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${TISSUE_FILE}")
else
  TISSUES=("${DEFAULT_TISSUES[@]}")
fi
mapfile -t CHR_LIST < <(expand_range_list "${CHROMOSOMES}")

[[ -f "${PREDICT_SCRIPT}" ]] || { echo "Missing Predict.py: ${PREDICT_SCRIPT}" >&2; exit 2; }
[[ "${VCF_TEMPLATE}" == *'{chr}'* ]] || { echo "VCF_TEMPLATE must contain {chr}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

for tissue in "${TISSUES[@]}"; do
  model_name="$(render_template "${MODEL_PATTERN}" '{tissue}' "${tissue}")"
  model_path="${MODEL_DIR}/${model_name}"
  tissue_out="${OUTPUT_ROOT}/${tissue}"
  [[ -f "${model_path}" ]] || { echo "Missing model database: ${model_path}" >&2; exit 2; }
  mkdir -p "${tissue_out}"

  for chr in "${CHR_LIST[@]}"; do
    vcf_path="$(render_template "${VCF_TEMPLATE}" '{chr}' "${chr}")"
    [[ -f "${vcf_path}" ]] || { echo "Missing chromosome VCF: ${vcf_path}" >&2; exit 2; }

    command=(
      "${PYTHON_BIN}" "${PREDICT_SCRIPT}"
      --model_db_path "${model_path}"
      --model_db_snp_key "${MODEL_DB_SNP_KEY}"
      --vcf_genotypes "${vcf_path}"
      --vcf_mode "${VCF_MODE}"
      --prediction_output "${tissue_out}/en_chr${chr}_prediction.txt"
      --prediction_summary_output "${tissue_out}/en_chr${chr}_summary.txt"
      --verbosity 9
      --throw
    )
    if [[ -n "${LIFTOVER_CHAIN}" ]]; then
      [[ -f "${LIFTOVER_CHAIN}" ]] || { echo "Missing liftover chain: ${LIFTOVER_CHAIN}" >&2; exit 2; }
      command+=(--liftover "${LIFTOVER_CHAIN}")
    fi
    if [[ -n "${MAPPING_TEMPLATE}" ]]; then
      command+=(--on_the_fly_mapping METADATA "${MAPPING_TEMPLATE}")
    fi

    echo "[PrediXcan] tissue=${tissue} chr=${chr}"
    "${command[@]}"
  done
done

echo "Predicted-expression outputs: ${OUTPUT_ROOT}"
