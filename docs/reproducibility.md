# Reproducibility notes

## Nested cross-validation

All supervised choices must happen inside the outer-training partition. For
traditional classifiers, imputation and scaling are pipeline steps evaluated
inside every inner split. For DiseaseCapsule, the same transforms are fitted
manually inside every inner split. The final transform is refitted using the
complete outer-training partition before evaluating the outer test fold.

The genotype workflow runs GWAS and per-gene PCA independently for each outer
fold. Unique outer-training samples fit GWAS, imputation, and PCA. The outer
test fold is transformed only after those objects are fitted.

PrediXcan applies external genotype-to-expression weights without phenotype
labels. Any subsequent case/control gene filtering is supervised and must be
repeated using only each outer-training fold. The human workflow therefore
keeps inference/merge separate from fold-specific Wilcoxon selection.

For crops, per-gene held-out metrics are computed from an expression-observed
training/test split. After that evaluation, the model is refitted on all
expression-observed samples before predicting the target population; target
samples never determine imputation values, lambda, or model coefficients.

## Duplicated split indices

Some historical disease split files contain duplicated positive-class rows as
an upsampling mechanism. Duplicates can cross inner folds and create leakage.
The new disease workflows deduplicate outer-training indices by default and
use class weights. `--keep-duplicate-train-rows` exists only for historical
result reproduction.

## Prediction and attribution outputs

Every benchmark exports sample-level outer-fold predictions. Human attribution
scripts export raw model-derived rankings separately from any prior-aware
ranking. A wet-lab prior must never be described as an unbiased genome-wide
discovery score.
