# 04. Crop predicted gene expression

`cis_elastic_net_expression.R` trains one cis elastic-net model per expressed
gene from crop genotype and observed expression data. It reports held-out R2,
MSE, and Spearman correlation, then refits each gene model on all
expression-observed samples before predicting the target population.

The same R implementation supports Rice18K and Maize1404 through explicit
GFF chromosome prefixes, expression delimiters, FAM ID columns, chromosome
lists, and PLINK-prefix templates.

```bash
DATA_DIR=/secure/Rice18K \
GFF_FILE=/secure/annotations/osa1_r7.all_models.gff3 \
GENE_EXP=/secure/Rice18K/all_Rpkm_samplename_match.txt \
PREDICT_FAM=/secure/Rice18K/LD_NAM_Magic.fam \
PLINK_PREFIX_TEMPLATE='/secure/Rice18K/chr{chr}_imputed' \
CHR_PREFIX=Chr CHROMOSOMES=1-12 MAX_JOBS=4 \
bash run_chromosomes_parallel.sh

Rscript merge_chromosome_outputs.R \
  --result_dir ../../outputs/crop_predicted_expression/Rice18K \
  --chromosomes 1-12 \
  --output_expression ../../outputs/crop_predicted_expression/Rice18K_PE.tsv \
  --output_metrics ../../outputs/crop_predicted_expression/Rice18K_metrics.tsv
```

For Maize1404, use `CHR_PREFIX=chr`, `CHROMOSOMES=1-10`, and
`GENE_EXP_SEP=,`. `FAM_ID_COLUMN=1` retains the historical FID convention;
set it to `2` when expression columns and prediction samples use IID.

The workflow requires R packages `optparse`, `data.table`, `progress`,
`plink2R`, `glue`, and `glmnet`, plus PLINK 1.9. The plotting utility accepts
merged chromosome metric directories and produces publication-format metric
histograms.
