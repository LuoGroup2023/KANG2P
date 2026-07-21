# 01. Genotype preprocessing

`run_nested_pipeline.sh` has three modes: `qc`, `gene-pca`, and `all`. The
`all` mode applies paper-style PLINK QC and then performs GWAS, ANNOVAR mapping,
training-only imputation, and per-gene PCA separately in every outer fold.

Required external tools are PLINK 1.9 and ANNOVAR with
`humandb/hg19_refGene.txt`.

```bash
SOURCE_PREFIX=/secure/ALS/cohort \
SPLIT_DIR=/secure/ALS/cv_splits_5fold \
ANNOVAR_DIR=/opt/annovar \
OUT_ROOT=/work/ALS_preprocessing \
PLINK_BIN=/opt/plink \
bash run_nested_pipeline.sh all --skip-existing
```

The output for each fold is a PKL bundle containing ordered training/test
matrices, labels, sample IDs, selected SNPs, and feature names. A TSV dimension
report summarizes all folds.
