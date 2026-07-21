# 05. Comparison experiments

This module groups the disease and conventional baseline experiments that use
the same predefined outer folds:

```text
comparison_experiments/
├── disease_capsule/  DiseaseCapsule nested-CV classifier
└── traditional_ml/   LR/RF/SVM/AdaBoost plus plant GS baselines
```

For human disease comparisons, `run_disease_baselines.sh` sends the same
aligned G/PE/PP inputs, labels, folds, seed, and duplicate-index policy to both
families:

```bash
bash run_disease_baselines.sh \
  --features pkl:/secure/ALS/genotype.pkl \
  --features tsv:/secure/ALS/predicted_expression.tsv \
  --labels pkl:/secure/ALS/labels.pkl \
  --split-dir /secure/ALS/cv_splits_5fold \
  --output-root ../../outputs/comparisons/als_g_pe \
  --prefix als_g_pe
```

Use `CAPSNET_*` variables for DiseaseCapsule tuning, `ML_GRID_JOBS` and
`ML_RF_JOBS` for parallelism, and `ML_METHODS="lr rf svm adaboost"` to select
traditional classifiers. Plant regression baselines remain under
`traditional_ml/` because they use ID-based folds and quantitative traits.
