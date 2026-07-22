# KANG2P

> **KANG2P enables interpretable genotype-to-phenotype prediction through genetically anchored molecular-state learning.**

KANG2P is a research framework for learning phenotypes from inherited genetic
variation together with molecular states predicted from that variation. It
connects genotype preprocessing, genetically predicted gene expression,
predicted protein abundance, nested-cross-validation benchmarks, DualKAN
gated fusion, and gradient-based interpretation in one auditable workflow.

The central idea is to represent an individual with complementary layers:

- **G — genotype:** inherited variation summarized at SNP or gene level;
- **PE — predicted expression:** a genetically regulated transcriptomic state;
- **PP — predicted proteome:** a genetically regulated protein-abundance state;
- **Y — phenotype:** a disease label or quantitative agronomic trait.

Rather than treating PE and PP as unrelated auxiliary tables, KANG2P treats
them as molecular-state projections anchored to the genotype. This makes it
possible to compare direct genetic prediction with prediction mediated through
biologically interpretable gene and protein features. Genetic anchoring does
not by itself establish causality; it defines the provenance of the molecular
features and reduces ambiguity about how they were derived.

## KANG2P workflow

![KANG2P workflow](KANG2P_workflow_v4.png)

There are two related analysis paths:

1. **Human disease prediction:** preprocess genotype, infer tissue-specific
   expression and/or protein abundance, compare DiseaseCapsule and traditional
   classifiers using identical outer folds, and interpret G or G+PE models.
2. **Crop genomic prediction:** learn cis-genetic expression models from
   observed crop expression, predict PE in the target population, compare
   conventional genomic-prediction baselines, and train DualKAN on G+PE.

## Repository organization

```text
workflows/
├── 01_genotype_preprocessing/     PLINK QC + fold-specific GWAS/Gene-PCA
├── 02_human_predicted_expression/ PrediXcan/MetaXcan tissue-expression inference
├── 03_predicted_proteome/          PP alignment, QC and benchmark entry point
├── 04_crop_predicted_expression/   cis elastic-net expression prediction in R
├── 05_comparison_experiments/      shared outer-fold baseline comparisons
│   ├── disease_capsule/            DiseaseCapsule nested CV
│   └── traditional_ml/             disease classifiers + plant GS baselines
├── 06_dualkan_gated_fusion/        G+PE DualKAN crop prediction
└── 07_human_gradient_analysis/     human G and G+PE attribution analyses

docs/                               data contracts, provenance and reproducibility
requirements/                       Python and external dependency notes
```

## Workflow guide

| Workflow | Main inputs | Main outputs | Purpose |
|---|---|---|---|
| [01. Genotype preprocessing](workflows/01_genotype_preprocessing/README.md) | PLINK BED/BIM/FAM, predefined outer folds, ANNOVAR | QC genotype, fold-specific GWAS SNPs, Gene-PCA matrices | Construct G without fitting supervised transformations on held-out samples |
| [02. Human predicted expression](workflows/02_human_predicted_expression/README.md) | chromosome VCFs, MetaXcan/PrediXcan tissue databases | tissue-specific sample-by-gene PE matrices | Project inherited variation into a genetically regulated transcriptomic state |
| [03. Predicted proteome](workflows/03_predicted_proteome/README.md) | per-sample protein predictions, optional model allow-list | aligned and QC-filtered PP matrix, transform metadata | Prepare a genetically predicted protein state for downstream comparison |
| [04. Crop predicted expression](workflows/04_crop_predicted_expression/README.md) | crop PLINK files, GFF3, observed expression, target FAM | predicted crop PE, gene-level R2/MSE/SCC | Learn cis-genetic expression models and transfer them to target crop samples |
| [05. Comparison experiments](workflows/05_comparison_experiments/README.md) | aligned G/PE/PP, labels or traits, predefined folds | fold predictions, selected parameters, metrics | Compare DiseaseCapsule and conventional baselines under matched evaluation |
| [06. DualKAN gated fusion](workflows/06_dualkan_gated_fusion/README.md) | crop G, crop PE, quantitative traits | fold predictions/metrics, tuning history, top-k feature counts, checkpoints | Learn nonlinear G+PE representations and modality interactions |
| [07. Human gradient analysis](workflows/07_human_gradient_analysis/README.md) | trained human models, G and PE matrices | feature/gene rankings from gradient, IG, LIME and weights | Trace phenotype predictions back to genotype components and genes |

## What “genetically anchored molecular-state learning” means

### 1. Genotype is the anchor

All derived molecular features retain a defined path back to inherited
variation. Human PE is generated with external tissue-specific genetic
weights. Crop PE is generated with cis-SNP elastic-net models. PP is consumed
as a genetically predicted abundance matrix with explicit sample and model
metadata.

### 2. Molecular states provide interpretable intermediate representations

PE and PP are organized by genes or proteins, whereas a raw genotype matrix
may contain hundreds of thousands or millions of variants. These intermediate
representations make it possible to ask whether phenotype signal is carried
by direct genotype features, genetically regulated molecular features, or
their interaction.

### 3. Fusion remains phenotype-predictive

KANG2P supports both simple early concatenation and structured multimodal
learning. The DualKAN model uses independent G and PE encoders, learnable
modality gates, interaction features, reconstruction losses, and either a
spline or Fourier KAN prediction head.

### 4. Interpretation is attached to the trained prediction model

Human analysis aggregates gradients, gradient × input, integrated gradients,
encoder weights, and grouped local-surrogate scores from component features to
genes. Raw model-derived rankings are kept separate from prior-aware rankings.

## Evaluation design

KANG2P uses predefined outer folds so every model can be evaluated on the same
held-out individuals. The implementation follows these rules:

- The outer-test fold is used only for final evaluation.
- Supervised SNP selection and Gene-PCA are repeated within each outer fold.
- Missing-value imputation and scaling are fitted using training samples only.
- Traditional classifiers place imputation and scaling inside
  `GridSearchCV`, giving each inner fold independent preprocessing statistics.
- DiseaseCapsule refits preprocessing separately within every inner-training
  split and again on the complete outer-training fold.
- Historical split files containing duplicated upsampled rows are deduplicated
  by default; class weighting handles imbalance without allowing duplicate
  individuals to cross inner folds.
- Every comparison exports sample-level outer-fold predictions, fold metrics,
  and selected hyperparameters.

Human PrediXcan inference does not use phenotype labels. However, any
case/control filtering of predicted genes is supervised and must be performed
independently inside each outer-training fold. The human PE workflow therefore
separates inference and merging from the optional fold-specific Wilcoxon step.

For crop PE, the R workflow first holds out expression-observed samples to
estimate per-gene R2, MSE, and Spearman correlation. It then refits the chosen
gene model using all expression-observed samples before predicting the target
population. Target samples do not determine imputation statistics, lambda, or
model coefficients.

See [reproducibility notes](docs/reproducibility.md) for additional details.

## Installation

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/all.txt
```

Core Python dependencies include NumPy, pandas, SciPy, scikit-learn,
matplotlib, PyTorch, Optuna, and XGBoost.

### External command-line tools

- PLINK 1.9 for genotype QC, extraction, and crop cis-SNP modeling;
- ANNOVAR with the hg19 `refGene` database for human gene annotation;
- MetaXcan/PrediXcan and compatible tissue model databases for human PE;
- an appropriate genome-build liftover chain when model and VCF builds differ.

### R dependencies

Crop predicted expression requires `optparse`, `data.table`, `progress`,
`plink2R`, `glue`, and `glmnet`. Traditional plant genomic prediction also
requires the `G2P` package.

External software, genetic model databases, annotations, model weights, and
controlled cohort data are not vendored in this repository.

## Quick start

### 1. Fold-safe genotype preprocessing

```bash
SOURCE_PREFIX=/secure/ALS/cohort \
SPLIT_DIR=/secure/ALS/cv_splits_5fold \
ANNOVAR_DIR=/opt/annovar \
OUT_ROOT=outputs/ALS/genotype_preprocessing \
bash workflows/01_genotype_preprocessing/run_nested_pipeline.sh all
```

The `all` mode runs PLINK QC followed by fold-specific GWAS, gene annotation,
training-derived missing-value imputation, and Gene-PCA. Use `qc` or
`gene-pca` to run one stage.

### 2. Human tissue-specific predicted expression

```bash
PREDICT_SCRIPT=/opt/MetaXcan/software/Predict.py \
MODEL_DIR=/secure/predixcan/elastic_net_models \
VCF_TEMPLATE='/secure/ALS/chr{chr}.phased.vcf.gz' \
LIFTOVER_CHAIN=/secure/chains/hg19ToHg38.over.chain.gz \
OUTPUT_ROOT=outputs/human_predicted_expression \
bash workflows/02_human_predicted_expression/run_predixcan.sh

Rscript workflows/02_human_predicted_expression/merge_predixcan_outputs.R \
  --input_dir outputs/human_predicted_expression/Brain_Cortex \
  --output outputs/human_predicted_expression/Brain_Cortex.tsv
```

The launcher includes the brain tissues used by the historical analysis. A
custom one-column `TISSUE_FILE` can be supplied without modifying the script.

### 3. Predicted-proteome preparation

```bash
python workflows/03_predicted_proteome/prepare_predicted_proteome.py \
  --prediction-table /secure/AD/raw_protein_predictions.tsv \
  --fit-sample-ids /secure/AD/outer_training_ids.txt \
  --standardize \
  --output outputs/predicted_proteome/ad_pp.tsv
```

When standardization is enabled, `--fit-sample-ids` is mandatory. The output
includes transformation parameters and a record of proteins removed for
missingness or near-zero variance.

### 4. Crop cis-genetic expression prediction

```bash
DATA_DIR=/secure/Rice18K \
GFF_FILE=/secure/annotations/osa1_r7.all_models.gff3 \
GENE_EXP=/secure/Rice18K/all_Rpkm_samplename_match.txt \
PREDICT_FAM=/secure/Rice18K/LD_NAM_Magic.fam \
PLINK_PREFIX_TEMPLATE='/secure/Rice18K/chr{chr}_imputed' \
CHR_PREFIX=Chr CHROMOSOMES=1-12 MAX_JOBS=4 \
bash workflows/04_crop_predicted_expression/run_chromosomes_parallel.sh
```

Use `CHR_PREFIX=chr`, `CHROMOSOMES=1-10`, and `GENE_EXP_SEP=,` for Maize1404.

### 5. Matched disease comparison

```bash
bash workflows/05_comparison_experiments/run_disease_baselines.sh \
  --features pkl:/secure/ALS/genotype_gene_pca.pkl \
  --features tsv:/secure/ALS/predicted_expression.tsv \
  --labels pkl:/secure/ALS/labels.pkl \
  --split-dir /secure/ALS/cv_splits_5fold \
  --output-root outputs/comparisons/als_g_pe \
  --prefix als_g_pe
```

This single entry point sends identical inputs and fold settings to
DiseaseCapsule and LR/RF/SVM/AdaBoost. Use `ML_METHODS`, `ML_GRID_JOBS`, and
`ML_RF_JOBS` for traditional classifiers and `CAPSNET_*` variables for the
capsule-network search.

For a PP-only comparison:

```bash
PP_TABLE=outputs/predicted_proteome/ad_pp.tsv \
LABEL_PKL=/secure/AD/labels.pkl \
SPLIT_DIR=/secure/AD/cv_splits_5fold \
bash workflows/03_predicted_proteome/run_pp_nested_cv.sh
```

### 6. DualKAN gated fusion

```bash
DATA_ROOT=/secure/plant \
TASK_LIST=Rice18K:Grain_yield \
GPU_LIST=0 \
HEAD_TYPE=fourier \
N_TRIALS=12 \
bash workflows/06_dualkan_gated_fusion/run_all_crops.sh
```

Remove `TASK_LIST` to run the configured Maize1404, Rice1495, and Rice18K trait
suite. Set `HEAD_TYPE=spline` to use spline KAN layers.

### 7. Human model interpretation

```bash
python workflows/07_human_gradient_analysis/als_gpe_dualbranch_interpretability.py \
  --g-pkl /secure/ALS/genotype_gene_pca.pkl \
  --pe-txt /secure/ALS/predicted_expression.tsv \
  --dualbranch-checkpoint /secure/checkpoints/als_g_pe.pt \
  --out-dir outputs/human_gradient/ALS/g_pe
```

## Input contracts

### Human disease matrices

- PKL feature bundles contain a samples-by-features DataFrame/array and may
  contain binary labels as their second item.
- TSV/CSV inputs use `IID`, then `FID`, then the first nonnumeric column as the
  sample identifier.
- Repeated `--features` arguments concatenate already aligned modalities.
- Disease outer-fold files contain zero-based row indices.

### Crop matrices

- Genotype and PE tables are samples by features with sample ID in column 1.
- Phenotype tables contain sample ID and one quantitative trait.
- Crop outer-fold files contain sample IDs rather than row positions.
- Crop expression-training tables are genes by observed samples.

Detailed layouts are documented in [data contracts](docs/data_layout.md).

## Output contracts

Depending on the workflow, KANG2P writes:

- fold-specific processed genotype or molecular-state matrices;
- sample-level outer-fold predictions and class probabilities;
- per-fold metrics and across-fold summaries;
- selected hyperparameters and feature indices;
- crop gene-expression prediction R2, MSE, SCC, and lambda values;
- PP imputation/scaling metadata and removed-protein reports;
- DualKAN checkpoints, tuning histories, fold predictions, metrics, and top-k
  feature counts;
- feature-level and gene-level attribution rankings.

Generated data, logs, checkpoints, caches, and result directories are ignored
by Git.

## Dataset references

The workflows do not download cohort data automatically. The original study
used or referenced the following resources:

- ALS data from [Project MinE](https://www.projectmine.com), with controlled
  dbGaP study accession `phs003146.v1.p1`;
- The data of AD was downloaded from dbGaP Study `phs000168.v2.p2`;
- the Rice18K/NAM variation resource on
  [Figshare](https://figshare.com/articles/dataset/NAM_variations/19166475);
- the Rice1495 hybrid-line resource from
  [CropGS](https://iagr.genomics.cn/CropGS);
- the maize CUBIC resource referenced by
  [G2P-env](https://github.com/G2P-env/G2P).

Users are responsible for obtaining the appropriate approvals, respecting the
source terms, and harmonizing sample identifiers and genome builds before
running the workflows.

## Data and reproducibility boundaries

Controlled human genotype data, sample-level molecular matrices, model
checkpoints, external genetic-weight databases, and generated predictions must
not be committed.

The audited workspace contains a historical 1,473-protein AD prediction matrix
and its downstream consumers, but not the upstream pQTL/protein-weight scoring
program or weight files. The repository therefore provides the reproducible
alignment, QC, transformation, and comparison boundary for PP without
inventing an unsupported upstream process. A complete external reproduction
must record the scoring tool version, weight source, genome build, allele
harmonization, and failed-model allow-list.

Human PE likewise depends on externally obtained MetaXcan/PrediXcan model
databases and compatible genome-build resources.

## Provenance

The mapping from audited source files to the reorganized workflows is recorded
in [docs/code_map.md](docs/code_map.md). New experiments should use the
numbered workflows and their documented input/output contracts.

## Intended use

KANG2P is research software for reproducible method development and evaluation.
Human disease predictions and feature rankings are not clinical diagnoses and
must not be interpreted as evidence of causality without independent genetic,
functional, and experimental validation.

Questions and reproducible bug reports can be submitted through this
repository's [GitHub Issues](https://github.com/LuoGroup2023/KANG2P/issues).

## License

See [LICENSE](LICENSE).
