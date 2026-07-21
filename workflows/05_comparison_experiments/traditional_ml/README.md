# Traditional ML baselines

`classification_nested_cv.py` evaluates logistic regression, random forest,
linear SVM, and AdaBoost for disease classification. Imputation and scaling
are inside the scikit-learn pipeline searched by inner CV.

`genomic_prediction_nested_cv.R` evaluates G2P regression methods such as
BayesC, BL, BRR, RKHS, RRBLUP, LASSO, SPLS, SVR, and RFR using predefined
outer folds. `tree_regression_nested_cv.py` provides Optuna-tuned RF/XGBoost
regression. The tree runner fits missing-value medians separately inside each
inner-training split and refits them on the full outer-training split.

The tree runner reads a sample-by-feature TSV directly when no valid cache is
present. Add `--create_cache` on the first run to save a reusable float32
matrix beside the TSV (useful for large plant data sets).

```bash
bash run_classification_benchmarks.sh \
  --features pkl:/secure/AD/genotype.pkl \
  --split-dir /secure/AD/cv_splits_5fold \
  --output-dir ../../../outputs/traditional_ml/ad_g \
  --prefix ad_g
```

Use `ML_GRID_JOBS` and `ML_RF_JOBS` to control nested parallelism.
