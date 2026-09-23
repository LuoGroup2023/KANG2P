# Data layout and contracts

Data paths are configurable. The following layout matches script defaults and
keeps protected data outside version control.

```text
data/
├── ALS/
│   ├── 02_merged_all_QCedSNPs.{bed,bim,fam}
│   ├── cv_splits_5fold/
│   │   ├── outer_fold_1_train_IDs.txt
│   │   └── outer_fold_1_test_IDs.txt
│   └── ...
├── human/ALS/
│   ├── genotype_gene_pca.pkl
│   ├── chr1.phased.vcf.gz ... chr22.phased.vcf.gz
│   ├── predicted_expression.tsv
│   └── predicted_proteome.tsv
└── plant/
    ├── Maize1404/
    ├── Rice1495/
    └── Rice18K/
checkpoints/
└── human/ALS/
outputs/
demo_data/
└── plant/Rice18K/                 synthetic reviewer fixture
```

## Human predicted-expression inputs

Human inference requires chromosome VCFs plus external MetaXcan/PrediXcan
tissue model databases. The VCF pattern must contain `{chr}`. Per-chromosome
outputs are merged only after verifying identical sample identifiers. Genome
build liftover chains and model databases remain outside Git.

## Predicted proteome input

The PP preparation workflow accepts one or more per-sample prediction tables.
It exports `FID`, `IID`, and protein columns. When standardization is enabled,
`--fit-sample-ids` is mandatory so transformation statistics are not estimated
from held-out samples.

## Crop predicted-expression inputs

The crop R workflow consumes chromosome PLINK files, a GFF3 annotation, a
genes-by-samples observed-expression table, and a FAM file defining the target
prediction samples. Rice and maize conventions are selected explicitly with
chromosome prefix, delimiter, and FAM ID-column options.

## Disease feature inputs

Feature PKLs must be tuples/lists whose first item is a samples-by-features
array or DataFrame. The second item may contain binary labels. Tabular inputs
are TSV/CSV matrices; `IID`, then `FID`, then a nonnumeric first column is used
as the sample ID. Repeated `--features` inputs must already have identical
sample and label order.

Outer-fold files contain zero-based row indices, one per line. The scripts
reject out-of-range indices and train/test overlap.

## Plant feature inputs

Each crop directory contains `X.txt` for genotype, `Exp.txt` or `PE.txt` for
predicted expression, phenotype tables, and ID-based outer-fold files. The
first column is the sample ID; remaining columns are numeric features.
The same contract is illustrated by the bundled synthetic fixture under
[`demo_data/plant/Rice18K`](../demo_data/plant/Rice18K).

## Files excluded from Git

PLINK binaries, sample-level omics matrices, checkpoints, result directories,
logs, caches, and external ANNOVAR databases are ignored. Small synthetic test
fixtures may be placed under `tests/fixtures/`.
