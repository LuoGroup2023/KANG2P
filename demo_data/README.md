# KANG2P demo data

This directory contains a small, deterministic synthetic dataset for checking
the DualKAN phenotype-prediction workflow without downloading controlled or
external biological data.

The `plant/Rice18K` fixture contains 60 synthetic samples:

- `X.txt`: 20 genotype features coded as 0, 1, or 2;
- `PE.txt`: 10 genetically anchored predicted-expression features;
- `Grain_yield.txt`: one synthetic quantitative trait;
- `outer_fold_*_train_IDs.txt` and `outer_fold_*_test_IDs.txt`: five
  non-overlapping outer folds.

The values are simulated and intentionally signal-rich so that the pipeline can
be exercised quickly. They are not study data and must not be used to draw
biological conclusions. See the repository root `README.md` for the reviewer
demo command.

