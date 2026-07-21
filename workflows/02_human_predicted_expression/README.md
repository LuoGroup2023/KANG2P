# 02. Human predicted gene expression

This workflow applies tissue-specific MetaXcan/PrediXcan weight databases to
human chromosome VCFs, merges the per-chromosome predictions, and optionally
selects disease-associated predicted-expression features within each outer
training fold.

```bash
PREDICT_SCRIPT=/opt/MetaXcan/software/Predict.py \
MODEL_DIR=/secure/predixcan/elastic_net_models \
VCF_TEMPLATE='/secure/ALS/chr{chr}.phased.vcf.gz' \
LIFTOVER_CHAIN=/secure/chains/hg19ToHg38.over.chain.gz \
OUTPUT_ROOT=../../outputs/human_predicted_expression \
bash run_predixcan.sh

Rscript merge_predixcan_outputs.R \
  --input_dir ../../outputs/human_predicted_expression/Brain_Cortex \
  --output ../../outputs/human_predicted_expression/Brain_Cortex.tsv
```

`MODEL_PATTERN`, `CHROMOSOMES`, `TISSUE_FILE`, `MODEL_DB_SNP_KEY`, and
`MAPPING_TEMPLATE` are configurable environment variables. MetaXcan itself,
GTEx/other tissue model databases, chain files, and controlled VCFs are
external assets and are not committed.

`select_training_differential_expression.R` reproduces the historical Wilcoxon
feature-selection step in a fold-safe form. Supply explicit outer-training
sample IDs and run it separately for every outer fold; never select genes on
the full cohort before nested cross-validation.
